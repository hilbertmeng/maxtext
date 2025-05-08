import functools
import subprocess
import requests
import time
import re
import argparse
import logging
import os

# 日志
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_bearer():
    return (
        subprocess.check_output("gcloud auth print-access-token", shell=True)
        .decode("utf-8")
        .strip()
    )


def check_tpu():
    project = "llm-tpu"
    headers = {
        "Authorization": f"Bearer {get_bearer()}",
    }
    response = requests.get(
        f"https://tpu.googleapis.com/v2alpha1/projects/{PROJECT}/locations/{TPU_ZONE}/nodes/{TPU_NAME}",
        headers=headers,
    )
    return response.json()


def wait_til():
    ret = check_tpu()
    if "error" in ret:
        logging.info(f"Error Status: 0")
        return ret, 0
    if ret["state"] in ["CREATING"]:
        logging.info(f"Unnormal Status: 1")
        return ret, 0
    elif ret["state"] in ["READY"]:
        logging.info(f"Normal Status: 2")
        return ret, 1
    else:
        logging.info(f"Unnormal Status: 3")
        return ret, 0


def create_tpu():
    start = time.time()
    create_num = 0
    request_num = 0
    while create_num < NUM:
        logging.info(
            f"\n\nStart {request_num}th try create {TPU_NAME}...  take time: {time.time() - start}s"
        )
        if 'v5p' in TPU_NAME or 'v6e' in TPU_NAME:
            command = f"gcloud alpha compute tpus queued-resources create {QUEUED_RESOURCE_ID} \
                        --node-id {TPU_NAME} \
                        --project {PROJECT} \
                        --zone {TPU_ZONE} \
                        --accelerator-type {TPU_TYPE} \
                        --runtime-version {VERSION} \
                        --service-account {SERVICE_ACCOUNT}  "
            if PREEMPTIBLE:
                command += " --best-effort"
        else:
            command = f"gcloud alpha compute tpus tpu-vm create {TPU_NAME}  \
                        --zone={TPU_ZONE} \
                        --accelerator-type={TPU_TYPE} \
                        --version={VERSION} \
                        --project={PROJECT}  \
                        --scopes=https://www.googleapis.com/auth/cloud-platform"
            if PREEMPTIBLE:
                command += " --preemptible"

        if "v4"  in TPU_NAME:
           # command += " --network v4-vpc-network"
            command += " --network=default --subnetwork=v4-vpc-network"

        logging.info(f"command:\n{command}")
        r = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
        # 获取返回值code
        return_code = r.returncode
        if int(return_code) == 0:
            logging.info(
                f"Create {TPU_NAME} success, take time: {time.time() - start}s !!!"
            )
            create_num += 1
        time.sleep(1)
        request_num += 1
    logging.info("Start to get tpu worder=0 ip.")
    response = subprocess.run(
        f"gcloud alpha compute tpus tpu-vm describe {TPU_NAME} --zone={TPU_ZONE} --project={PROJECT}",
        stdout=subprocess.PIPE,
        shell=True,
    )
    output = response.stdout.decode("utf-8")
    ip_addresses = re.findall(r"ipAddress: ([\d.]+)", output)
    if 'v5p' in TPU_NAME or 'v6e' in TPU_NAME:
        head_ip = '0.0.0.0'
        status = check_status(t=10, n=None)
    else:
        head_ip = ip_addresses[0]
    return head_ip


def install():
    # install
    start = time.time()
    scp_command = f"gcloud compute tpus tpu-vm scp {INSTALL_FILE} {TPU_NAME}:~/  --zone={TPU_ZONE}  --worker=all  --project={PROJECT}"
    parent_zone = TPU_ZONE.rsplit("-", maxsplit=1)[0]
    logging.info(f"TPU_ZONE: {TPU_ZONE}")
    assert parent_zone in ["us-east1", "us-west4", "us-central2", "us-central1", "europe-west4", "us-east5"], logging.info(
        f"parent_zone: {parent_zone}"
    )
    install_command = f'gcloud compute tpus tpu-vm ssh {TPU_NAME} --zone={TPU_ZONE}  --worker=all  --project={PROJECT} --command="bash {INSTALL_FILE} {parent_zone}"'
    logging.info(f"scp_command: {scp_command}")
    logging.info(f"install_command: {install_command}")
    commands = [scp_command, install_command]
    scp_code = subprocess.run(scp_command, stdout=subprocess.PIPE, shell=True)
    logging.info(f"scp_code: {scp_code.returncode}")
    install_code = subprocess.run(install_command, stdout=subprocess.PIPE, shell=True)
    logging.info(f"install_code: {install_code.returncode}")
    return scp_code.returncode, install_code.returncode


def train(train_script):
    logging.info(f".........Start train.......")
    train_command = f'bash {train_script} {TPU_TYPE}'
    train_code = subprocess.run(train_command, stdout=subprocess.PIPE, shell=True)
    logging.info(f'train_code.returncode: {train_code.returncode}')
    return train_code.returncode


def del_tpu():
    logging.info(f"Start del old tpu: {TPU_NAME}...")
    del_command = f"gcloud compute tpus tpu-vm delete {TPU_NAME} --zone={TPU_ZONE} --quiet --project {PROJECT}"
    subprocess.run(del_command, stdout=subprocess.PIPE, shell=True)
    if 'v5p' in  TPU_NAME or 'v6e' in TPU_NAME:
        extr_del_command = f"echo 'Y' | gcloud alpha compute tpus queued-resources delete {QUEUED_RESOURCE_ID} --zone={TPU_ZONE}  --project {PROJECT}"
        logging.info(f'extr_del_command: {extr_del_command}')
        subprocess.run(extr_del_command, stdout=subprocess.PIPE, shell=True)


def check_status(t=3, n=None):
    status = []
    # 检查三次
    i = 0
    while True:
        state, code = wait_til()
        logging.info(f"Check {i}th...\nStatus: {status}\n State: {state}\n")
        status.append(int(code))
        i += 1
        if n is None and int(code) > 0:
            break

        if n is not None and i >= n:
            break
        time.sleep(t)

    logging.info(f"Status: {status}")
    return status


def find_log_path():
    find_log_cmd = f'find {REMOTE_DIR.rstrip("/")}/ -type f -name "*.train.log"'
    remote_find_log_cmd = f'gcloud compute tpus tpu-vm ssh {TPU_NAME} --zone={TPU_ZONE} --worker=0 --command="{find_log_cmd}"  --project={PROJECT}'
    log_code = subprocess.run(remote_find_log_cmd, stdout=subprocess.PIPE, shell=True, text=True)
    log_name = os.path.basename(log_code.stdout.strip().split('\n')[-1]) # 只取最后一个log
    log_path = os.path.join(REMOTE_DIR, log_name)
    return log_path


def process_log(break_num, log_path):
    logging.info(f'Start upload log file to bucket.')
    # 每隔一段时间将训练日志上传到bucket
    upload_log_cmd = f'gcloud compute tpus tpu-vm ssh {TPU_NAME} --zone={TPU_ZONE} --worker=0 --command="gsutil cp {log_path} gs://newproject-1-llm_base_models_europe-west4/logs/break-{break_num}/"  --project={PROJECT}'
    subprocess.run(upload_log_cmd, stdout=subprocess.PIPE, shell=True, text=True)

    remote_tail_log_cmd = f'gcloud compute tpus tpu-vm ssh {TPU_NAME} --zone={TPU_ZONE} --worker=0 --command="tail -n 10 {log_path}"  --project={PROJECT}'
    tail_code = subprocess.run(remote_tail_log_cmd, stdout=subprocess.PIPE, shell=True, text=True)
    lines = tail_code.stdout.strip().split('\n')
    stop = False
    for line in lines:
        logging.info(line)
        step = extract_train_step(line)
        print(f'step: {step}')
        if step > TOTAL_STEPS:
            logging.info(f'Current step exceed total steps: {TOTAL_STEPS}, start to del tpu.')
            del_tpu() # 删除完成才会往后执行
            logging.info(f'Delete tpu success.')
            stop = True
            break
    return stop


def extract_train_step(line_log):
    pat = re.compile('completed step: (\d+), steps/s')
    # completed step: 388, steps/s:
    step = pat.findall(line_log)
    return int(step[0]) if step else -1


def run():
    break_num = -1 # 记录被强占的次数
    enter_train = 0
    start = time.time()
    while True:
        status = check_status(t=1, n=CHECK_NUM)
        if sum(status) == 0:
            break_num += 1
            enter_train = 0
            if DELETE: # 如果已经存在的tpu，是不会再进来的
                del_tpu()
            # create tpu, 成功后才会返回
            head_ip = create_tpu()
            logging.info(f"Head_ip: {head_ip}")

            if INSTALL_FILE:
                start = time.time()
                _, install_code = install()
                if install_code == 0:
                    logging.info(
                        f"Install environment success... take time: {time.time() - start}s"
                    )
                else:
                    logging.info(f"Install environment fail... please check code again...")

            if TRAIN_SCRIPT:
                train_code = train(TRAIN_SCRIPT)
                # if train_code != 0:
                #     logging.info(f"Train fail......please check code again......")
        if enter_train == 0:
            log_path = find_log_path()
            enter_train = 1
            train_code = train(TRAIN_SCRIPT)

        stop = process_log(break_num, log_path) # upload log and tail -n 10 log message

        if not CHECK or stop:
            logging.info(f'Stop is true, now exit code.')
            exit()
        logging.info(f"Waiting {CHECK_FRE}s to run next loop")
        time.sleep(CHECK_FRE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create tpu script")

    parser.add_argument(
        "--project",
        type=str,
        default="llm-tpu",
        choices=["llm-tpu", "colorful-aia", "ntpu-413714", "newproject-1-451205"],
        help="Tpu project",
    )
    parser.add_argument("--tpu_name", type=str, default=None, help="Tpu name")
    # 主要参数1
    parser.add_argument("--type", type=str, help="Tpu type, such as v3-8, v3-32...")
    parser.add_argument(
        "--zone",
        type=str,
        default="us-east1-d",
        choices=["us-east1-d", "us-west4-a", "us-central2-b", "us-central1-a", "europe-west4-b", "us-east5-a", "europe-west4-a"],
        help="tpu zone",
    )
    # 主要参数2
    parser.add_argument("--suffix", type=str, default="0", help="Tpu suffix")
    # 主要参数3
    parser.add_argument(
        "-p",
        "--preemptible",
        action="store_true",
        default=False,
        help="Whether tpu is preemptible",
    )
    # 主要参数4
    parser.add_argument(
        "-inf",
        "--install_file",
        type=str,
        default="",
        help="Tpu evironment install's shell file",
    )
    parser.add_argument(
        "-oni",
        "--only_install",
        action="store_true",
        default=False,
        help="Whether only to install environment",
    )
    parser.add_argument(
        "-ont",
        "--only_train",
        action="store_true",
        default=False,
        help="Whether only to t train",
    )
    parser.add_argument(
        "-del",
        "--delete",
        action="store_true",
        default=False,
        help="Whether del the old tpu",
    )
    parser.add_argument("--num", type=int, default=1, help="tpu number")
    parser.add_argument(
        "-c",
        "--check",
        action="store_true",
        default=False,
        help="Whether loop to check tpu status",
    )
    parser.add_argument(
        "-chf",
        "--check_fre",
        type=int,
        default=60,
        help="How long time to check tpu status",
    )
    parser.add_argument(
        "-chn", 
        "--check_num", type=int, default=3, help="Check tpu status frequcy"
    )
    parser.add_argument(
        "-trsp",
        "--train_script",
        type=str,
        default=None,
        help="Tpu train script path",
    )
    parser.add_argument("-v", "--version", type=str, default=None, help="Tpu vm type")
    parser.add_argument("-ts", "--total_steps", type=int, default=0, help="Total train step numbers")

    args = parser.parse_args()
    REMOTE_DIR = '/home/lishengping/projects/maxtext'
    HEALTH_STATE = {"state": "READY", "health": "HEALTHY"}

    TPU_TYPE = args.type  # tpu 类型
    SUFFIX = args.suffix
    TPU_NAME = (
        f"llm-jax-{TPU_TYPE}-{SUFFIX}" if args.tpu_name is None else args.tpu_name
    )
    TPU_ZONE = args.zone  # 区域
    PREEMPTIBLE = args.preemptible  # 抢占式
    PROJECT = args.project  # 项目id
    NUM = args.num  # 申请NUM个tpu，名字后缀从SUFFIX开始累加。 type: int
    DELETE = args.delete  # 是否删除之前同名的tpu, type: bool
    CHECK_FRE = args.check_fre  # 每隔CHECK_FRE秒检查tpu状态。type: int
    CHECK_NUM = args.check_num  # 连续检查CHECK_NUM次tpu状态，因为有的时候检查一次不一定准确。type: int
    CHECK = args.check  # 是否循环检查tpu的状态
    ONLY_INSTALL = args.only_install  # 仅仅安装环境
    INSTALL_FILE = args.install_file  # 构建tpu环境的shell文件路径, type: str
    ONLY_TRAIN = args.only_train  # 仅训练
    TRAIN_SCRIPT = args.train_script  # 训练命令
    VERSION = args.version
    TOTAL_STEPS = int(args.total_steps) + 50

    logging.info(f'TOTAL_STEPS: {TOTAL_STEPS}')

    if VERSION is None:
        if "v3" in TPU_NAME:
            VERSION = "tpu-vm-base"
            assert TPU_ZONE in ["us-east1-d", "us-central1-a", "europe-west4-a"]
        elif "v4" in TPU_NAME:
            VERSION = "tpu-vm-tf-2.10.0-pod-v4"
            assert TPU_ZONE in ["us-central2-b", 'us-central1-a']
        elif "v5p" in TPU_NAME:
            VERSION = "v2-alpha-tpuv5"
            QUEUED_RESOURCE_ID = TPU_NAME
            assert TPU_ZONE in ["us-east5-a", "europe-west4-b"]
            SERVICE_ACCOUNT = "887571727717-compute@developer.gserviceaccount.com"
            SERVICE_ACCOUNT = "626151558586-compute@developer.gserviceaccount.com"
        elif "v5" in TPU_NAME:
            VERSION = "v2-alpha-tpuv5-lite"
            assert TPU_ZONE in ["us-west4-a"]
        elif "v6e" in TPU_NAME:
            VERSION = "v2-alpha-tpuv6e"
            QUEUED_RESOURCE_ID = TPU_NAME
            assert TPU_ZONE in ["europe-west4-a"]
            SERVICE_ACCOUNT = "887571727717-compute@developer.gserviceaccount.com"
            SERVICE_ACCOUNT = "626151558586-compute@developer.gserviceaccount.com"
        else:
            raise ValueError(f"TPU_TYPE: {TPU_TYPE} not in ‘v3,v4,v5’...")

    if ONLY_INSTALL:
        install()
    else:
        run()


# Usage：
# 仅申请tpu v3-8
# python create_tpu.py --type v3-8 --suffix 0  -p

# 申请tpu v3-8 并安装环境
# python create_tpu.py --type v3-8 --suffix 0  -p -inf install_new.sh

# 申请tpu v5litepod-16 并安装环境
# python create_tpu.py --type v5litepod-16 --zone us-west4-a --suffix 0 -inf install_new.sh -p

# 删除老的tpu, 申请新的tpu v3-8 并安装环境
# python create_tpu.py --type v3-8 --suffix 0 -inf install_new.sh -p --del

# 申请tpu v3-8 并安装环境,并训练
# python create_tpu.py --type v3-8 --suffix 0 -inf install_new.sh -p -trc configs/8-test-llama.json --del

# python create_tpu.py --type v3-8 --suffix 0 -inf install_new.sh -p -trsp train_script --del


# 删除老的tpu, 申请新的 tpu v3-8 并安装环境,并训练
# python create_tpu.py --type v3-8 --suffix 0 -inf install_new.sh -p -trc configs/8-test-llama.json --del

# 删除老的tpu, 申请新的 tpu v3-8 并安装环境,并训练, 循环检查tpu是否被抢占
# python create_tpu.py --type v3-8 --suffix 0 -inf install_new.sh -p -trc configs/8-test-llama.json --del --check

# 仅安装环境
# python create_tpu.py --type v3-8 --suffix 0  -oni -inf install_new.sh

# 仅训练
# python create_tpu.py --type v3-8 --suffix 0 -ont --trc configs/8-test-llama.json

# 监听。一旦发现被抢占。执行申请，安装，训练。trsp: 训练脚本路径，ont: 仅训练，ts：总的训练步数，一旦发现达到这个步数，将自动删除tpu并退出
#  python3 auto_train.py --type v5p-8 --suffix 10 -inf newproject_install_maxtext.sh --zone europe-west4-b --project newproject-1-451205 -p -del -c -ts 2000 -trsp /home/lishengping/lsp/train.sh
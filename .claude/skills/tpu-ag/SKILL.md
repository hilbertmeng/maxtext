---
name: tpu-ag
description: Run commands on remote server tpu-ag via SSH. Use when the user asks to do anything on tpu-ag, the remote TPU orchestration host, or needs to read/edit files on the remote machine.
---

# tpu-ag Remote Server

Host `tpu-ag` → `lishengping@35.186.124.92`. xd's remote workspace: `/home/lishengping/xd/projects/`.

## SSH Multiplexed Connection

Use a persistent master connection to avoid re-authentication on each command.

### Setup (run once per session)

```bash
ssh -o StrictHostKeyChecking=accept-new -fNM -S /tmp/ssh-tpu-ag-xd.sock tpu-ag
```

### Run commands

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag "command"
```

### Check / tear down

```bash
ssh -S /tmp/ssh-tpu-ag-xd.sock -O check tpu-ag
ssh -S /tmp/ssh-tpu-ag-xd.sock -O exit tpu-ag
```

## Editing Remote Files

Local file tools (Read, Write, StrReplace) cannot access remote files. Use:

1. **Read**: `ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag "cat /path/to/file"`
2. **Write**: `ssh -S /tmp/ssh-tpu-ag-xd.sock tpu-ag "cat > /path/to/file << 'EOF' ... EOF"`
3. **Complex edits**: `scp -o 'ControlPath=/tmp/ssh-tpu-ag-xd.sock'` locally, edit, copy back.

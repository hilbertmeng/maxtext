"""Render the configured BAM/Transformer architecture without importing JAX.

Run: python experiments/bam_llama2_medium/architecture_shared_read_llf/render_architecture.py
Produces editable SVG, vector PDF and PNG previews alongside this script.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
CONFIG = 'BamLlama2MediumV2C256LocalFetchC8SharedReadLLFScan'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'mathtext.fontset': 'dejavusans'})
C = dict(ink='#182B43', mute='#52647A', blue='#2267A8', bluebg='#EFF6FF',
         teal='#087F83', tealbg='#EAF8F5', amber='#A46114', amberbg='#FFF5E6',
         line='#D7E0E8', white='#FFFFFF', bg='#F7F9FC', purple='#7151A1')

class Canvas:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w/100, h/100), dpi=100, facecolor=C['bg'])
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0,w), ylim=(h,0)); self.ax.axis('off')
        self.labels = []

    def text(self, x,y,s,size=13,color='ink',weight='normal',ha='left'):
        t = self.ax.text(x,y,s,fontsize=size,color=C.get(color,color),weight=weight,
                        ha=ha,va='center',linespacing=1.5,zorder=5)
        self.labels.append(t)
        return t

    def box(self,x,y,w,h,title,sub='',kind='blue',size=13):
        self.ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=12',
            facecolor=C.get(kind+'bg',C['white']),edgecolor=C.get(kind,C['line']),linewidth=1.25,zorder=2))
        multi = '\n' in sub
        title_y = y + (max(20, h*.28) if multi else h/2-(12 if sub else 0))
        self.text(x+w/2,title_y,title,size,weight='bold',ha='center')
        if sub: self.text(x+w/2,y+(h*.69 if multi else h/2+17),sub,size-2,'mute',ha='center')

    def panel(self,x,y,w,h,kicker,title,subtitle=''):
        self.ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=18',
            facecolor=C['white'],edgecolor=C['line'],linewidth=1,zorder=0))
        self.text(x+24,y+30,kicker,11,'teal','bold')
        self.text(x+24,y+64,title,20,weight='bold')
        if subtitle: self.text(x+24,y+98,subtitle,11,'mute')

    def arrow(self,points,color='blue',dash=False,lw=1.8):
        codes=[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1)
        self.ax.add_patch(FancyArrowPatch(path=MPath(points,codes),arrowstyle='-|>',
            mutation_scale=13,color=C[color],lw=lw,linestyle='--' if dash else '-',zorder=3))

    def line(self,points,color='line',lw=1):
        self.ax.plot(*zip(*points),color=C[color],lw=lw,zorder=1)

    def save(self,name):
        # Force a render and verify every text label stays within the canvas.
        self.fig.canvas.draw()
        renderer=self.fig.canvas.get_renderer()
        for label in self.labels:
            b=label.get_window_extent(renderer)
            if b.x0 < 0 or b.y0 < 0 or b.x1 > self.w+1 or b.y1 > self.h+1:
                raise ValueError(f'Clipped label: {label.get_text()} {b}')
        for ext in ('svg','pdf','png'):
            self.fig.savefig(OUT/f'{name}.{ext}',dpi=150 if ext=='png' else 100,
                             facecolor=self.fig.get_facecolor())


def main_figure():
    c=Canvas(1800,1670)
    c.text(40,42,'BAM as a Transformer add-on',29,weight='bold')
    c.text(40,84,CONFIG,13,'mute')
    c.text(40,115,'24 layers  /  D = 1024  /  16 MHA heads × 64  /  SwiGLU width = 2816',13,'mute')
    for x,col,label in [(1110,'blue','Transformer'),(1320,'teal','BAM read / route'),(1580,'amber','BAM write')]:
        c.line([(x,116),(x+25,116)],col,3); c.text(x+34,116,label,11,col)

    c.panel(35,150,1730,300,'01 / MODEL & DECODER','Two streams through depth: residual vectors + per-token matrix state')
    # Residual path through a decoder.
    items=[(65,135,'Embedding','tokens → '+r'$h_0$'),(265,145,'RMSNorm',r'$x_\ell$'),
           (455,245,'MHA + BAM','expanded below'),(750,80,'+',''),
           (880,150,'RMSNorm',''),(1080,155,'SwiGLU','1024 → 2816 → 1024'),
           (1285,80,'+',''),(1460,265,'Final norm + LM head','after layer 24')]
    for x,w,t,s in items: c.box(x,287,w,65,t,s,size=12)
    for a,b in zip(items,items[1:]): c.arrow([(a[0]+a[1],320),(b[0],320)])
    c.arrow([(230,320),(230,270),(790,270),(790,287)])
    c.arrow([(850,320),(850,259),(1325,259),(1325,287)])
    c.text(260,390,r'$M_\ell\quad(M_0=0)$',13,'teal')
    c.arrow([(410,390),(470,390)],'teal')
    c.arrow([(438,390),(438,363),(480,363),(480,352)],'teal')
    c.box(470,365,230,52,'BAM state update',kind='amber',size=12)
    c.arrow([(578,352),(578,365)],'amber')
    c.arrow([(700,390),(900,390)],'teal')
    c.text(915,390,r'$M_{\ell+1}$ : 32 × 32 per token',13,'teal')
    c.text(1270,390,'[ L → L → F ] × 8',19,'teal','bold')
    c.text(65,430,'Repeat the decoder from RMSNorm to the second residual add; each layer has its own parameters.',11,'mute')

    def layer_panel(ox,mode):
        is_l=mode=='L'
        c.panel(ox,480,850,890,'02 / LOCAL LAYER' if is_l else '03 / FETCH LAYER',
                'L  ·  Shared read into V and O' if is_l else 'F  ·  MHA routes matrix retrieval',
                'Layers 1, 2, 4, 5, …, 22, 23' if is_l else 'Layers 3, 6, 9, …, 24')
        a=ox+30; b=ox+480
        c.text(a,620,r'$x_\ell=\mathrm{RMSNorm}(h_\ell)$',14,'blue')
        c.text(b,620,r'$M_\ell(t)\in\mathbb{R}^{32\times32}$',15,'teal')
        c.box(a,655,340,90,'Q / K projections + RoPE',r'$Q=\mathrm{RoPE}(xW_Q)+\Delta Q$'+'\n'+r'$K=\mathrm{RoPE}(xW_K)+\Delta K$',size=12)
        c.box(b,655,335,90,'Local Q/K read','Full M · rank-1 head routing\nKeys, gates & head weights from x',kind='teal',size=12)
        c.arrow([(b+165,636),(b+165,655)],'teal')
        c.arrow([(b,700),(a+340,700)],'teal')
        c.text(a+386,680,r'$\Delta Q,\Delta K$',10,'teal',ha='center')
        c.arrow([(a+170,745),(a+170,790)])
        c.box(a,790,340,85,'Causal attention weights',r'$A_n=\mathrm{softmax}(Q_nK_n^T/\sqrt{64}+\mathrm{mask})$',size=12)
        c.box(b,790,335,70,'Address-axis compression',r'$S_\ell=M_\ell P_c\quad (32\times8)$',kind='teal',size=12)
        # Same M input branches to full Q/K read and compressed view.
        c.arrow([(b+344,623),(b+353,623),(b+353,778),(b+168,778),(b+168,790)],'teal')
        c.text(a,912,r'$V^0=xW_V$',12,'blue')
        c.arrow([(a+170,875),(a+170,935)])
        if is_l:
            c.box(a,935,340,80,'MHA value aggregation',r'$Y_n(t)=\sum_{s\leq t}A_n(t,s)\,[V_n^0(s)+\Delta V_n(s)]$',size=11)
            c.box(b,890,335,85,'One bilateral read',r'$R_n(t)=\mathrm{Read}(S_\ell(t),x_t)$'+'\nRMS keys; shared pre-gate result',kind='teal',size=12)
            c.arrow([(b+168,860),(b+168,890)],'teal')
            c.box(b,1010,150,65,r'$G_V(x)\odot R$',r'$\Delta V$',kind='teal',size=12)
            c.box(b+185,1010,150,65,r'$G_O(x)\odot R$',r'$\Delta O$',kind='teal',size=12)
            c.arrow([(b+168,975),(b+168,990),(b+75,990),(b+75,1010)],'teal')
            c.arrow([(b+168,990),(b+260,990),(b+260,1010)],'teal')
            c.arrow([(b,1042),(b-40,1042),(b-40,975),(a+340,975)],'teal')
            c.box(a,1120,340,75,'Fused head output',r'$O_n=Y_n+\Delta O_n$',size=13)
            c.arrow([(b+260,1075),(b+260,1157),(a+340,1157)],'teal')
            c.text(b+168,1230,'Shared read; independent V/O gates.',12,'teal',ha='center')
            c.text(b+168,1260,'Each gate has separate row / column factors.',10,'mute',ha='center')
        else:
            c.box(a,935,340,80,'MHA value aggregation',r'$Y_n(t)=\sum_{s\leq t}A_n(t,s)V_n^0(s)$',size=12)
            c.box(b,905,335,100,'Reuse MHA attention weights',r'$\beta_{ts}=\sum_n w_{tn}(x_t)A_n(t,s)$'+'\nSigned head mix; set '+r'$\beta_{tt}=1$',kind='teal',size=12)
            c.arrow([(a+340,833),(b-40,833),(b-40,955),(b,955)],'teal')
            c.box(b,1050,335,80,'Fetch compressed matrices',r'$\bar{S}_t=\sum_{s\leq t}\beta_{ts}S_\ell(s)$',kind='teal',size=12)
            c.arrow([(b+168,1005),(b+168,1050)],'teal')
            c.arrow([(b+335,825),(b+355,825),(b+355,1090),(b+335,1090)],'teal')
            c.box(b,1170,335,80,'Bilateral read + gate',r'$\Delta O_n=\mathrm{GatedRead}_n(\bar{S}_t,x_t)$',kind='teal',size=12)
            c.arrow([(b+168,1130),(b+168,1170)],'teal')
            c.box(a,1120,340,75,'Fused head output',r'$O_n=Y_n+\Delta O_n$',size=13)
            c.arrow([(b,1210),(b-40,1210),(b-40,1157),(a+340,1157)],'teal')
            c.text(b+168,1290,'One fetched matrix; 16 output read heads.',11,'teal',ha='center')
        c.arrow([(a+170,1015),(a+170,1120)])
        c.arrow([(a+170,1195),(a+170,1240)])
        c.box(a,1240,340,70,r'$\mathrm{Concat}(O_n)\,W_O$','→ residual add → RMSNorm → SwiGLU → add',size=11)
        c.arrow([(a,1157),(a-15,1157),(a-15,1335),(a+170,1335)],'amber')
        c.text(a+183,1335,'O → matrix write (below)',11,'amber')
    layer_panel(35,'L'); layer_panel(915,'F')

    c.panel(35,1395,1730,190,'04 / WRITE IN EVERY LAYER','The fused MHA + BAM head output becomes new matrix content')
    c.box(65,1480,340,75,'Data from fused head output',r'$u_n=\mathrm{RMS}(O_n[0:32])$',kind='amber',size=12)
    c.box(455,1480,385,75,'Address + write gate from x',r'$p_n=\mathrm{RMS}(P_{loc,n}(x)),\quad g_n=\sigma(W_{g,n}x+b_n)$',kind='amber',size=11)
    c.arrow([(405,1518),(430,1518),(430,1570),(895,1570),(895,1518),(925,1518)],'amber')
    c.arrow([(840,1518),(925,1518)],'amber')
    c.box(925,1480,445,75,'Per-token outer-product update',r'$M_{\ell+1}(t)=M_\ell(t)+\sum_{n=1}^{16}g_{tn}u_{tn}p_{tn}^{T}$',kind='amber',size=12)
    c.arrow([(1370,1518),(1430,1518)],'teal')
    c.text(1450,1502,'To the next layer',15,'teal','bold')
    c.text(1450,1532,'Full 32 × 32 state',12,'teal')
    c.text(40,1610,'Local = same-token BAM read. All 24 layers use global causal MHA. C256 = query tiling; C8 = compressed address width.',12,'mute')
    c.text(40,1642,'Layer scan executes [L,L,F] blocks with distinct layer parameters. Q/K additions are after RoPE; QK normalization is disabled.',11,'mute')
    return c


def details_figure():
    c=Canvas(1800,1290)
    c.text(40,44,'BAM read / write mechanics',29,weight='bold')
    c.text(40,88,'Companion to the Transformer interaction diagram · same SharedRead LLF configuration',13,'mute')
    c.panel(35,125,850,450,'A / BILATERAL READ','One matrix, two directions')
    c.box(75,260,215,170,r'$Z\in\mathbb{R}^{32\times v}$','k / U = data\nv / P_loc = address',kind='teal',size=17)
    c.box(405,260,425,75,'Column read → data coordinates',r'$z_{col}=Z\,\widehat r_v(x)\in\mathbb{R}^{32}$',kind='teal',size=13)
    c.box(405,370,425,75,'Row read → address coordinates',r'$z_{row}=Z^T\widehat r_k(x)\in\mathbb{R}^{v}$',kind='teal',size=13)
    c.arrow([(290,298),(405,298)],'teal');c.arrow([(290,408),(405,408)],'teal')
    c.text(75,485,r'$\widehat r=\mathrm{RMS}(r)$'+'   ·   row and column keys normalized separately',13,'teal')
    c.text(75,522,'Read keys, routing weights and sigmoid gates are functions of x.',12,'mute')
    c.text(75,551,'The persistent matrix stream has no whole-matrix RMS normalization.',11,'mute')

    c.panel(915,125,850,450,'B / EXACT WIDTHS','Full state and compressed read view')
    c.box(950,255,280,75,r'$M_\ell(t):32\times32$','persistent across layers',kind='teal',size=15)
    c.box(1390,255,335,75,r'$S_\ell(t)=M_\ell(t)P_c$','32 × 8; learned P_c: 32 → 8',kind='teal',size=14)
    c.arrow([(1230,293),(1390,293)],'teal')
    c.text(1260,269,'compress',11,'teal')
    c.arrow([(1090,330),(1090,372)],'teal');c.arrow([(1557,330),(1557,372)],'teal')
    c.box(950,372,280,85,'Local Q/K','32 data + 32 address\n= 64 coordinates per head',kind='teal',size=12)
    c.box(1390,372,335,85,'Local O/V and fetched O','32 data + 8 address + 24 zeros\n= 64 coordinates per head',kind='teal',size=12)
    c.text(950,500,'Compression changes only the read view; every write updates full M.',12,'mute')
    c.text(950,535,'No learned 8 → 32 decoder: compressed row results occupy slots 32:40.',11,'mute')

    c.panel(35,600,850,395,'C / LOCAL QK & SHARED O/V','What is shared, and what is independent?')
    c.text(75,720,'Local Q/K: one basis per read side, separately for Q and K.',13,'ink','bold')
    c.text(75,761,r'$\Delta Q_n=[a^Q_{n,col}z^Q_{col}\ ;\ a^Q_{n,row}z^Q_{row}]$',17,'teal')
    c.text(75,801,'Here z includes side gates; signed, RMS-normalized head weights route it.',11,'mute')
    c.text(75,837,'K uses the same construction with its own keys, gates and head weights.',11,'mute')
    c.line([(75,865),(845,865)])
    c.text(75,900,'Local O/V: 16 read heads share one pre-gate result between destinations.',12,'ink','bold')
    c.text(75,941,r'$\Delta V_n=G_{V,n}(x)\odot R_n,\quad\Delta O_n=G_{O,n}(x)\odot R_n$',17,'teal')
    c.text(75,975,r'$G_{V,n},G_{O,n}$'+' each have row / column factors '+r'$2\sigma(\cdot)$'+'. No cross-layer weight tying.',11,'mute')

    c.panel(915,600,850,395,'D / FETCH ROUTE & WRITE','Reuse attention for transport; write locally')
    c.text(950,715,r'$w_t=\mathrm{RMS}(W_{mix}x_t)/\sqrt{16}$',18,'teal')
    c.text(950,770,r'$\beta_{ts}=\sum_n w_{tn}A_n(t,s)\ (s<t),\qquad\beta_{tt}=1$',16,'teal')
    c.text(950,823,r'$\bar S_t=S_\ell(t)+\sum_{s<t}\beta_{ts}S_\ell(s)$',16,'teal')
    c.text(950,858,'Signed route; causal / segment mask inherited from MHA; no second softmax.',11,'mute')
    c.line([(950,883),(1725,883)])
    c.text(950,914,r'$P_{loc}(x)=W_{up}\,\mathrm{GELU}(W_{down}x)+b$',16,'amber')
    c.text(950,947,'1024 → 256 → (16 × 32); one address factor per MHA head.',12,'mute')
    c.text(950,975,'Write uses O before W_O; retention = 1; data and address factors use RMS.',11,'mute')

    c.panel(35,1020,1730,220,'E / READING THE TWO STREAMS','M carries depth history; MHA supplies token-to-token communication')
    c.text(75,1140,'L layer:  M(t) → Q/K, V and O at token t; injected V is then transported by ordinary MHA.',15,'teal')
    c.text(75,1180,'F layer:  the existing MHA map transports S(s) to t; a query-conditioned read converts the fetched matrix into O.',14,'teal')
    c.text(75,1218,'Both:  fused O(t) supplies write data, while x(t) supplies the address and write gate; the FFN updates only the residual stream.',12,'mute')
    c.text(40,1270,'Source: current exp.py inheritance + BamAttention + FusionDecoderLayer + initial_bam_matrix. Batch axes and numerical epsilon terms omitted.',10,'mute')
    return c


if __name__=='__main__':
    first=main_figure();first.save('bam_transformer_interaction')
    second=details_figure();second.save('bam_read_write_details')
    with PdfPages(OUT/'bam_architecture.pdf') as pdf:
        pdf.savefig(first.fig);pdf.savefig(second.fig)
    sources=['MaxText/exp.py','MaxText/layers/attentions.py','MaxText/layers/fusion.py',
             'MaxText/layers/models.py','MaxText/configs/base.yml']
    manifest={'config':CONFIG,'basis':'current working-tree implementation; not a historical runtime reconstruction',
              'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}}
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Rendered architecture figures in {OUT}')

"""
Passos Magicos · Painel de Indicadores e Preditor de Risco
Datathon POSTECH — Fase 5.   Rodar:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import core

#  paleta 
NAVY="#0E1B27"; PANEL="#16283A"; LINE="#25384B"
TEAL="#0E9E8E"; AMBER="#E8A13A"; BLUE="#4F9CE8"
GOOD="#35C08A"; WARN="#F2B14A"; BAD="#E86A5C"; NEUTRAL="#DDE7EE"
TXT="#E6EDF3"; SUB="#9FB3C4"
SERIES=["#4FB0C6","#35C08A","#E8A13A","#E86A5C","#9A7BD8","#5A8DEE","#C6A15B"]
PEDRA_ORDER=["QUARTZO","AGATA","AMETISTA","TOPAZIO"]
PEDRA_COLORS={"QUARTZO":"#B9A7D9","AGATA":"#8FBF9F","AMETISTA":"#9A6ABF","TOPAZIO":"#E8A13A"}
PEDRA_LABEL={"QUARTZO":"Quartzo","AGATA":"Ágata","AMETISTA":"Ametista","TOPAZIO":"Topázio"}

st.set_page_config(page_title="Passos Magicos · Indicadores & Risco",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stMetric {{ font-family:'Inter',sans-serif; }}
.block-container {{ padding-top:1.4rem; padding-bottom:2rem; max-width:1300px; }}
#MainMenu, footer {{ visibility:hidden; }}
.hero {{ background:linear-gradient(120deg,#0E9E8E 0%,#12456A 55%,#0E1B27 100%);
         border-radius:16px; padding:26px 32px; margin-bottom:18px;
         box-shadow:0 8px 30px rgba(0,0,0,.35); }}
.hero .kick {{ color:#FCE9C8; letter-spacing:3px; font-size:12px; font-weight:700; }}
.hero h1 {{ color:#fff; margin:6px 0 4px 0; font-size:30px; font-weight:800; letter-spacing:-.5px; }}
.hero p  {{ color:#DCEBF0; margin:0; font-size:14px; }}
.sec {{ display:flex; align-items:center; gap:10px; margin:8px 0 6px 0; }}
.sec .bar {{ width:5px; height:22px; background:{AMBER}; border-radius:3px; }}
.sec h3 {{ color:{TXT}; margin:0; font-size:19px; font-weight:700; }}
.kpi {{ background:{PANEL}; border:1px solid {LINE}; border-radius:12px; padding:14px 18px; min-height:92px; }}
.kpi .l {{ color:{SUB}; font-size:12.5px; font-weight:600; text-transform:uppercase; letter-spacing:.5px; }}
.kpi .v {{ font-size:31px; font-weight:800; line-height:1.15; margin-top:2px; }}
.insight {{ background:rgba(232,161,58,.08); border-left:3px solid {AMBER};
            padding:10px 14px; border-radius:8px; color:#CBD8E2; font-size:13px; line-height:1.5; margin-top:6px; }}
/* --- pergunta (pagina Analises) --- */
.qh {{ display:flex; gap:14px; align-items:flex-start; margin:2px 0 2px 0; }}
.qh .qn {{ flex:0 0 auto; width:34px; height:34px; border-radius:9px; background:{TEAL};
           color:#fff; font-weight:800; font-size:16px; display:flex; align-items:center; justify-content:center; }}
.qh .qt {{ color:{TXT}; font-size:17px; font-weight:700; line-height:1.2; }}
.qh .qp {{ color:{SUB}; font-size:13px; margin-top:2px; }}
/* --- barra lateral / navegacao --- */
section[data-testid="stSidebar"] {{ background:{PANEL}; }}
.brand {{ color:{TXT}; font-size:18px; font-weight:800; letter-spacing:.3px; margin:2px 0 2px 0; }}
.brand span {{ color:{AMBER}; }}
.navlabel {{ color:{SUB}; font-size:11px; font-weight:700; text-transform:uppercase;
             letter-spacing:1.5px; margin:12px 0 4px 2px; }}
section[data-testid="stSidebar"] .stButton>button {{
    width:100%; text-align:left; justify-content:flex-start;
    border-radius:10px; font-weight:600; font-size:14.5px; padding:9px 14px; margin:2px 0;
    border:1px solid transparent; }}
section[data-testid="stSidebar"] .stButton>button[kind="secondary"] {{
    background:transparent; color:#C7D5E0; }}
section[data-testid="stSidebar"] .stButton>button[kind="secondary"]:hover {{
    background:rgba(255,255,255,.05); color:#fff; border-color:{LINE}; }}
section[data-testid="stSidebar"] .stButton>button[kind="primary"] {{
    background:{TEAL}; color:#fff; border-color:{TEAL}; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data(): return core.load_data("dados_consolidados.csv")

@st.cache_resource
def get_models(_df): return core.train_models(_df)   # apenas_constantes=True

df = get_data()
DFC = core.alunos_constantes(df)                      # base do notebook (constantes)
models, meta = get_models(df)
ANOS = sorted(df["Ano"].dropna().unique().tolist())

#  helpers 
def style(fig, h=330, legend=True):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=SUB, family="Inter", size=12),
                      margin=dict(t=34, b=8, l=8, r=8),
                      title=dict(text=""),
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=SUB)))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False, linecolor=LINE)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False, linecolor=LINE)
    if not legend: fig.update_layout(showlegend=False)
    return fig

def sec(title):
    st.markdown(f"<div class='sec'><div class='bar'></div><h3>{title}</h3></div>", unsafe_allow_html=True)

def insight(html):
    st.markdown(f"<div class='insight'>{html}</div>", unsafe_allow_html=True)

def kpi(col, label, value, color):
    col.markdown(f"<div class='kpi'><div class='l'>{label}</div><div class='v' style='color:{color}'>{value}</div></div>",
                 unsafe_allow_html=True)

def qhead(n, titulo, pergunta):
    st.markdown(f"<div class='qh'><div class='qn'>{n}</div><div><div class='qt'>{titulo}</div>"
                f"<div class='qp'>{pergunta}</div></div></div>", unsafe_allow_html=True)

def fmt_list(vals, casas=1, suf=""):
    """Rotulos explicitos: NaN vira '' (evita 'undefined'/'nan' no Plotly)."""
    out=[]
    for v in vals:
        out.append("" if (v is None or (isinstance(v,float) and np.isnan(v))) else f"{v:.{casas}f}{suf}")
    return out

def sem(kind, v):
    faixas={"inde":(7.0,6.0),"emfase":(40,25),"topazio":(20,12)}
    bom,med=faixas[kind]
    return GOOD if v>=bom else (WARN if v>=med else BAD)

def corr_word(r):
    a=abs(r)
    return "forte" if a>=0.6 else "moderada" if a>=0.4 else "fraca" if a>=0.2 else "muito fraca"

def trend_word(d):
    return "de alta" if d>0.05 else "de queda" if d<-0.05 else "estável"

def kpi_row(f):
    if len(f)==0:
        st.info("Sem dados para os filtros selecionados."); return
    inde=f["INDE"].mean(); emf=(f["Faixa_defasagem"]=="Em fase").mean()*100
    top=(f["Pedra"]=="TOPAZIO").mean()*100
    c=st.columns(4)
    kpi(c[0],"Alunos", f"{len(f):,}".replace(",","."), NEUTRAL)
    kpi(c[1],"INDE médio", f"{inde:.2f}", sem("inde",inde))
    kpi(c[2],"% em fase", f"{emf:.1f}%", sem("emfase",emf))
    kpi(c[3],"% Topázio", f"{top:.1f}%", sem("topazio",top))

def pedra_present(f, col):
    """Media por pedra, apenas pedras com dado (evita NaN -> 'undefined')."""
    g=f.dropna(subset=["Pedra"]).groupby("Pedra")[col].mean()
    pres=[p for p in PEDRA_ORDER if p in g.index and pd.notna(g[p])]
    return pres, [g[p] for p in pres]

#  hero 
st.markdown("""
<div class="hero">
  <div class="kick">ASSOCIAÇÃO PASSOS MÁGICOS · DATATHON PEDE 2022–2024</div>
  <h1>Painel de Indicadores &amp; Preditor de Risco</h1>
  <p>Desenvolvimento educacional dos alunos e previsão de risco de defasagem — de forma interativa.</p>
</div>
""", unsafe_allow_html=True)

#  navegacao (botoes) 
PAGES = ["Visão Geral", "Indicadores", "Análises", "Previsão de Risco"]
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]

st.sidebar.markdown("<div class='brand'>Passos <span>Mágicos</span></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='navlabel'>Navegação</div>", unsafe_allow_html=True)
for p in PAGES:
    if st.sidebar.button(p, key=f"nav_{p}",
                         type="primary" if st.session_state.page==p else "secondary"):
        st.session_state.page = p
        st.rerun()
page = st.session_state.page

# filtro Novo/Constante so nas paginas onde faz sentido
tipo = "Todos"
if page in ("Visão Geral", "Indicadores"):
    st.sidebar.markdown("<div class='navlabel'>Filtro de alunos</div>", unsafe_allow_html=True)
    tipo = st.sidebar.radio("Tipo", ["Todos", "Constantes", "Novos"], horizontal=True,
                            label_visibility="collapsed",
                            help="Constante = já estava no programa · Novo = ingressou naquele ano.")

st.sidebar.markdown("<hr style='border-color:#25384B'>", unsafe_allow_html=True)
st.sidebar.caption(f"Registros: **{len(df):,}**".replace(",", ".") + f"  ·  Anos: **{ANOS[0]}–{ANOS[-1]}**")
st.sidebar.caption(f"Modelos · RF AUC **{meta['auc_rf']:.2f}** · Rede Neural AUC **{meta['auc_nn']:.2f}**")
st.sidebar.caption(f"Modelo treinado com **{meta['n_ras_treino']}** alunos constantes.")

tipo_map={"Constantes":"Constante","Novos":"Novo"}
dfx = df if tipo=="Todos" else df[df["tipo_aluno"]==tipo_map[tipo]]

# = VISAO GERAL
if page == "Visão Geral":
    with st.expander("Filtros", expanded=False):=
        anos_sel = st.multiselect("Ano", ANOS, default=ANOS)
    f = dfx[dfx["Ano"].isin(anos_sel)] if anos_sel else dfx
    kpi_row(f); st.markdown("")

    sec("Volumetria de alunos — novos x constantes")
    with st.container(border=True):
        vol = df.groupby(["Ano","tipo_aluno"]).size().reset_index(name="Alunos")
        figv = px.bar(vol, x="Ano", y="Alunos", color="tipo_aluno", barmode="group",
                      color_discrete_map={"Novo":AMBER,"Constante":TEAL},
                      text=fmt_list(vol["Alunos"],0))
        figv.update_traces(textposition="outside"); figv.update_xaxes(type="category")
        figv.update_layout(legend_title="")
        st.plotly_chart(style(figv, 320), width='stretch')
        nv=df[df.tipo_aluno=="Novo"]; cs=df[df.tipo_aluno=="Constante"]
        n24=len(nv[nv.Ano==ANOS[-1]]); c24=len(cs[cs.Ano==ANOS[-1]])
        insight(f"Em <b>{ANOS[-1]}</b> houve <b>{n24}</b> registros de alunos novos e <b>{c24}</b> de constantes. "
                f"No total do período: <b>{len(cs):,}</b> constantes e <b>{len(nv):,}</b> novos.".replace(",","."))

    c1,c2 = st.columns(2)
    with c1:
        sec("Evolução da defasagem")
        with st.container(border=True):
            tab=(pd.crosstab(f["Ano"],f["Faixa_defasagem"],normalize="index")*100).round(1)
            tab=tab.reindex(columns=[c for c in ["Em fase","Moderada","Severa"] if c in tab.columns])
            lg=tab.reset_index().melt(id_vars="Ano",var_name="Faixa",value_name="pct")
            fig=px.bar(lg,x="Ano",y="pct",color="Faixa",
                       color_discrete_map={"Em fase":GOOD,"Moderada":WARN,"Severa":BAD},
                       text=fmt_list(lg["pct"],1,"%"))
            fig.update_layout(barmode="stack",yaxis_title="% de alunos",legend_title="")
            fig.update_xaxes(type="category")
            st.plotly_chart(style(fig,330), width='stretch')
            an=sorted(f["Ano"].unique()); t=(pd.crosstab(f["Ano"],f["Faixa_defasagem"],normalize="index")*100)
            ef0=t.loc[an[0]].get("Em fase",0); ef1=t.loc[an[-1]].get("Em fase",0)
            insight(f"A parcela de alunos <b>em fase</b> foi de {ef0:.1f}% em {an[0]} para <b>{ef1:.1f}%</b> em {an[-1]} — "
                    f"tendência {trend_word((ef1-ef0)/100)}.")
    with c2:
        sec("Indicadores médios no ciclo")
        with st.container(border=True):
            by=f.groupby("Ano")[["INDE","IDA","IEG"]].mean().reset_index()
            figl=go.Figure()
            for col,cl in [("INDE",BLUE),("IDA",TEAL),("IEG",GOOD)]:
                figl.add_trace(go.Scatter(x=by["Ano"],y=by[col],mode="lines+markers+text",name=col,
                              line=dict(color=cl,width=3),text=fmt_list(by[col],2),
                              textposition="top center",textfont=dict(color=cl)))
            figl.update_layout(yaxis_title="Nota média",legend_title=""); figl.update_xaxes(type="category")
            st.plotly_chart(style(figl,330), width='stretch')
            d_inde=by["INDE"].iloc[-1]-by["INDE"].iloc[0]
            insight(f"O INDE médio variou <b>{d_inde:+.2f}</b> no período ({by['INDE'].iloc[0]:.2f} → {by['INDE'].iloc[-1]:.2f}) — "
                    f"tendência {trend_word(d_inde)}.")

    sec("Distribuição por Pedra")
    with st.container(border=True):
        pt=(pd.crosstab(f["Ano"],f["Pedra"],normalize="index")*100).round(1)
        pt=pt.reindex(columns=[c for c in PEDRA_ORDER if c in pt.columns])
        lp=pt.reset_index().melt(id_vars="Ano",var_name="Pedra",value_name="pct")
        lp["Pedra"]=lp["Pedra"].map(PEDRA_LABEL)
        figp=px.bar(lp,x="Ano",y="pct",color="Pedra",text=fmt_list(lp["pct"],0,"%"),
                    color_discrete_map={PEDRA_LABEL[k]:v for k,v in PEDRA_COLORS.items()},
                    category_orders={"Pedra":[PEDRA_LABEL[p] for p in PEDRA_ORDER]})
        figp.update_layout(barmode="stack",yaxis_title="% de alunos",legend_title="")
        figp.update_xaxes(type="category")
        st.plotly_chart(style(figp,320), width='stretch')
        an=sorted(f["Ano"].unique())
        tp=(pd.crosstab(f["Ano"],f["Pedra"],normalize="index")*100)
        top0=tp.loc[an[0]].get("TOPAZIO",0); top1=tp.loc[an[-1]].get("TOPAZIO",0)
        insight(f"A fatia de alunos <b>Topázio</b> foi de {top0:.1f}% ({an[0]}) para <b>{top1:.1f}%</b> ({an[-1]}).")

# = INDICADORES
elif page == "Indicadores":
    fases=sorted(df["Fase_label"].dropna().unique(), key=lambda x:(x!="Alfa",x))
    with st.expander("Filtros", expanded=False):
        fsel=st.multiselect("Fase", fases, default=fases)
    f = dfx[dfx["Fase_label"].isin(fsel)] if fsel else dfx
    if len(fsel)<len(fases) and fsel:
        st.caption("Fases selecionadas: " + ", ".join(fsel))
    kpi_row(f); st.markdown("")
    st.caption("Um gráfico para cada indicador. Os textos abaixo de cada gráfico refletem os dados filtrados.")

    def card(col, titulo, fig, texto):
        with col:
            with st.container(border=True):
                st.markdown(f"**{titulo}**")
                if fig is None:
                    st.info("Sem dados para os filtros selecionados.")
                else:
                    st.plotly_chart(fig, width='stretch')
                    insight(texto)

    # linha 1: IAN, IDA
    a,b=st.columns(2)
    tab=(pd.crosstab(f["Ano"],f["Faixa_defasagem"],normalize="index")*100).round(1)
    tab=tab.reindex(columns=[c for c in ["Em fase","Moderada","Severa"] if c in tab.columns])
    lg=tab.reset_index().melt(id_vars="Ano",var_name="Faixa",value_name="pct")
    figIAN=px.bar(lg,x="Ano",y="pct",color="Faixa",color_discrete_map={"Em fase":GOOD,"Moderada":WARN,"Severa":BAD},
                  text=fmt_list(lg["pct"],0,"%"))
    figIAN.update_layout(barmode="stack",yaxis_title="% alunos",legend_title=""); figIAN.update_xaxes(type="category")
    an=sorted(f["Ano"].unique()); tt=(pd.crosstab(f["Ano"],f["Faixa_defasagem"],normalize="index")*100)
    tIAN=(f"Em fase: {tt.loc[an[0]].get('Em fase',0):.0f}% ({an[0]}) → <b>{tt.loc[an[-1]].get('Em fase',0):.0f}%</b> ({an[-1]}). "
          f"Severa em {an[-1]}: {tt.loc[an[-1]].get('Severa',0):.1f}%.")
    card(a,"IAN · Adequação de Nível (defasagem)",style(figIAN,300),tIAN)
    # IDA
    by=f.groupby("Ano")["IDA"].mean().reset_index()
    figIDA=px.line(by,x="Ano",y="IDA",markers=True,color_discrete_sequence=[TEAL],text=fmt_list(by["IDA"],2))
    figIDA.update_traces(textposition="top center",line_width=3); figIDA.update_layout(yaxis_title="IDA médio")
    figIDA.update_xaxes(type="category")
    dIDA=by["IDA"].iloc[-1]-by["IDA"].iloc[0]
    tIDA=f"IDA médio {trend_word(dIDA)}: <b>{dIDA:+.2f}</b> no período ({by['IDA'].iloc[0]:.2f} → {by['IDA'].iloc[-1]:.2f})."
    card(b,"IDA · Desempenho Acadêmico",style(figIDA,300,legend=False),tIDA)

    # linha 2: IEG, IAA
    a,b=st.columns(2)
    pres,vals=pedra_present(f,"IEG")
    if pres:
        figIEG=px.bar(x=[PEDRA_LABEL[p] for p in pres],y=vals,color=[PEDRA_LABEL[p] for p in pres],
                      color_discrete_map={PEDRA_LABEL[k]:v for k,v in PEDRA_COLORS.items()},
                      text=fmt_list(vals,1))
        figIEG.update_layout(yaxis_title="IEG médio",xaxis_title="",showlegend=False)
        figIEG.update_traces(textposition="outside")
        d=dict(zip(pres,vals))
        tIEG=(f"Engajamento cresce com a pedra: Quartzo <b>{d.get('QUARTZO',float('nan')):.1f}</b> vs Topázio <b>{d.get('TOPAZIO',float('nan')):.1f}</b>."
              if "QUARTZO" in d and "TOPAZIO" in d else "Engajamento médio por pedra.")
        figIEG=style(figIEG,300,legend=False)
    else:
        figIEG=None; tIEG=""
    card(a,"IEG · Engajamento (por Pedra)",figIEG,tIEG)
    # IAA scatter vs IDA
    ff=f.dropna(subset=["IAA","IDA"])
    if len(ff)>2:
        figIAA=px.scatter(ff,x="IAA",y="IDA",opacity=0.4,color_discrete_sequence=[BLUE])
        aa,bb=np.polyfit(ff["IAA"],ff["IDA"],1); xs=np.array([ff["IAA"].min(),ff["IAA"].max()])
        figIAA.add_trace(go.Scatter(x=xs,y=aa*xs+bb,mode="lines",line=dict(color=BAD,width=3),name="tend."))
        figIAA.update_layout(showlegend=False,xaxis_title="IAA",yaxis_title="IDA")
        rIAA=ff["IAA"].corr(ff["IDA"])
        tIAA=f"Correlação IAA × IDA = <b>{rIAA:.2f}</b> ({corr_word(rIAA)}): a autoavaliação {'quase não reflete' if abs(rIAA)<0.2 else 'reflete parcialmente'} o desempenho real."
        figIAA=style(figIAA,300,legend=False)
    else:
        figIAA=None; tIAA=""
    card(b,"IAA · Autoavaliação (× Desempenho)",figIAA,tIAA)

    # linha 3: IPS, IPP
    a,b=st.columns(2)
    ffp=f.dropna(subset=["IPS","Pedra"])
    presp=[p for p in PEDRA_ORDER if p in ffp["Pedra"].unique()]
    if presp:
        figIPS=px.box(ffp,x="Pedra",y="IPS",category_orders={"Pedra":presp},
                      color="Pedra",color_discrete_map=PEDRA_COLORS)
        figIPS.update_layout(showlegend=False,xaxis_title="",yaxis_title="IPS")
        figIPS.update_xaxes(ticktext=[PEDRA_LABEL[p] for p in presp],tickvals=presp)
        dips=dict(zip(*pedra_present(f,"IPS")))
        tIPS=(f"IPS médio por pedra varia pouco (Quartzo {dips.get('QUARTZO',float('nan')):.1f} · Topázio {dips.get('TOPAZIO',float('nan')):.1f}) — atua como base comportamental estável."
              if "QUARTZO" in dips and "TOPAZIO" in dips else "IPS por pedra — base comportamental estável.")
        figIPS=style(figIPS,300,legend=False)
    else:
        figIPS=None; tIPS=""
    card(a,"IPS · Psicossocial (por Pedra)",figIPS,tIPS)
    # IPP por categoria IAN
    def rot(v): return "Em fase" if v>=10 else ("Moderada" if v>=5 else "Severa")
    fp=f.dropna(subset=["IPP","IAN"]).copy(); fp["catIAN"]=fp["IAN"].apply(rot)
    order=["Em fase","Moderada","Severa"]; m=fp.groupby("catIAN")["IPP"].mean().reindex(order)
    figIPP=px.bar(x=order,y=m.values,color=order,color_discrete_map={"Em fase":GOOD,"Moderada":WARN,"Severa":BAD},
                  text=fmt_list(m.values,2))
    figIPP.update_layout(showlegend=False,xaxis_title="Adequação (IAN)",yaxis_title="IPP médio")
    figIPP.update_traces(textposition="outside")
    tIPP=f"IPP médio: em fase <b>{m.get('Em fase',float('nan')):.2f}</b> vs severa <b>{m.get('Severa',float('nan')):.2f}</b> — confirma a tendência na média, mas com grande dispersão individual."
    card(b,"IPP · Psicopedagógico (por nível de IAN)",style(figIPP,300,legend=False),tIPP)

    # linha 4: IPV, INDE
    a,b=st.columns(2)
    cols=[c for c in ["IEG","IDA","IPP","IAN","IAA","IPS"] if c in f.columns]
    cc=f[["IPV"]+cols].corr()["IPV"].drop("IPV").sort_values()
    figIPV=px.bar(x=cc.values,y=cc.index,orientation="h",text=fmt_list(cc.values,2),
                  color=cc.values,color_continuous_scale=["#5A6B78",TEAL])
    figIPV.update_layout(coloraxis_showscale=False,xaxis_title="Correlação com IPV",yaxis_title="")
    figIPV.update_traces(textposition="outside")
    topdrv=cc.sort_values(ascending=False).index[0]; topval=cc.sort_values(ascending=False).iloc[0]
    tIPV=f"O que mais acompanha o Ponto de Virada: <b>{topdrv}</b> (r={topval:.2f}). Engajar e estudar puxam a virada."
    card(a,"IPV · Ponto de Virada (drivers)",style(figIPV,300,legend=False),tIPV)
    # INDE por pedra
    presi,indv=pedra_present(f,"INDE")
    if presi:
        figIND=px.bar(x=[PEDRA_LABEL[p] for p in presi],y=indv,color=[PEDRA_LABEL[p] for p in presi],
                      color_discrete_map={PEDRA_LABEL[k]:v for k,v in PEDRA_COLORS.items()},text=fmt_list(indv,2))
        figIND.update_layout(showlegend=False,yaxis_title="INDE médio",xaxis_title="")
        figIND.update_traces(textposition="outside")
        tIND=f"INDE médio geral: <b>{f['INDE'].mean():.2f}</b>. Cresce de forma consistente do Quartzo ao Topázio."
        figIND=style(figIND,300,legend=False)
    else:
        figIND=None; tIND=""
    card(b,"INDE · Nota Geral (por Pedra)",figIND,tIND)

# = ANALISES (perguntas 1-11, exceto 9)
elif page == "Análises":
    d = DFC  # alunos constantes (recorte do notebook)
    st.caption(f"Respostas às perguntas de negócio do desafio, sobre os **{d['RA'].nunique()} alunos constantes** "
               f"(presentes nos três anos do ciclo). A pergunta 9 é atendida na página **Previsão de Risco**.")
    an=sorted(d["Ano"].unique()); a0,a1=an[0],an[-1]

    # Q1 IAN
    with st.container(border=True):
        qhead(1,"Adequação do nível (IAN)","Qual o perfil de defasagem dos alunos e como evolui ao longo dos anos?")
        tab=(pd.crosstab(d["Ano"],d["Faixa_defasagem"],normalize="index")*100).round(1)
        tab=tab.reindex(columns=[c for c in ["Em fase","Moderada","Severa"] if c in tab.columns])
        lg=tab.reset_index().melt(id_vars="Ano",var_name="Faixa",value_name="pct")
        fig=px.bar(lg,x="Ano",y="pct",color="Faixa",color_discrete_map={"Em fase":GOOD,"Moderada":WARN,"Severa":BAD},
                   text=fmt_list(lg["pct"],1,"%"))
        fig.update_layout(barmode="stack",yaxis_title="% de alunos",legend_title=""); fig.update_xaxes(type="category")
        st.plotly_chart(style(fig,320),width='stretch')
        t=(pd.crosstab(d["Ano"],d["Faixa_defasagem"],normalize="index")*100)
        insight(f"Alunos <b>em fase</b> saltaram de {t.loc[a0].get('Em fase',0):.1f}% ({a0}) para "
                f"<b>{t.loc[a1].get('Em fase',0):.1f}%</b> ({a1}); a defasagem <b>severa</b> caiu para "
                f"{t.loc[a1].get('Severa',0):.1f}%. Sinal claro de efetividade do acompanhamento.")

    # Q2 IDA
    with st.container(border=True):
        qhead(2,"Desempenho acadêmico (IDA)","O desempenho médio está melhorando, estagnado ou caindo?")
        by=d.groupby("Ano")["IDA"].mean().reset_index()
        fig=px.line(by,x="Ano",y="IDA",markers=True,color_discrete_sequence=[TEAL],text=fmt_list(by["IDA"],2))
        fig.update_traces(textposition="top center",line_width=3); fig.update_layout(yaxis_title="IDA médio")
        fig.update_xaxes(type="category")
        st.plotly_chart(style(fig,300,legend=False),width='stretch')
        dida=by["IDA"].iloc[-1]-by["IDA"].iloc[0]
        insight(f"IDA médio geral: {by['IDA'].iloc[0]:.2f} ({a0}) → <b>{by['IDA'].iloc[-1]:.2f}</b> ({a1}), "
                f"variação <b>{dida:+.2f}</b>. Houve pico em {an[1]} e recuo em {a1} — desempenho é a frente que ainda "
                f"exige atenção, apesar do avanço na adequação de nível.")

    # Q3 IEG  /  Q4 IAA
    c1,c2=st.columns(2)
    with c1:
        with st.container(border=True):
            qhead(3,"Engajamento (IEG)","Tem relação com desempenho (IDA) e ponto de virada (IPV)?")
            cc=d[["IEG","IDA","IPV"]].corr()["IEG"].drop("IEG")
            fig=px.bar(x=cc.index,y=cc.values,color=cc.index,
                       color_discrete_map={"IDA":TEAL,"IPV":BLUE},text=fmt_list(cc.values,2))
            fig.update_layout(showlegend=False,yaxis_title="Correlação com IEG",xaxis_title="",yaxis_range=[0,1])
            fig.update_traces(textposition="outside")
            st.plotly_chart(style(fig,280,legend=False),width='stretch')
            insight(f"Correlação <b>{corr_word(cc['IPV'])}</b> com o IPV (r={cc['IPV']:.2f}) e com o IDA (r={cc['IDA']:.2f}): "
                    f"quem se engaja tende a render mais e a alcançar a virada.")
    with c2:
        with st.container(border=True):
            qhead(4,"Autoavaliação (IAA)","A percepção do aluno é coerente com desempenho e engajamento?")
            cc=d[["IAA","IDA","IEG"]].corr()["IAA"].drop("IAA")
            fig=px.bar(x=cc.index,y=cc.values,color=cc.index,
                       color_discrete_map={"IDA":TEAL,"IEG":GOOD},text=fmt_list(cc.values,2))
            fig.update_layout(showlegend=False,yaxis_title="Correlação com IAA",xaxis_title="",yaxis_range=[0,1])
            fig.update_traces(textposition="outside")
            st.plotly_chart(style(fig,280,legend=False),width='stretch')
            insight(f"Correlação <b>muito fraca</b> (IDA r={cc['IDA']:.2f} · IEG r={cc['IEG']:.2f}): a forma como o aluno "
                    f"se enxerga nem sempre acompanha o desempenho real — sinal de fatores emocionais/contextuais.")

    # Q5 IPS (versao corrigida)
    with st.container(border=True):
        qhead(5,"Aspectos psicossociais (IPS)","Há padrões que antecedem quedas de desempenho ou engajamento?")
        ds=d.sort_values(["RA","Ano"]).copy()
        ds["IDA_next"]=ds.groupby("RA")["IDA"].shift(-1); ds["IEG_next"]=ds.groupby("RA")["IEG"].shift(-1)
        ds["queda_IDA"]=ds["IDA_next"]<ds["IDA"]; ds["queda_IEG"]=ds["IEG_next"]<ds["IEG"]
        ds["fx"]=pd.cut(ds["IPS"],[0,6,8,10],labels=["Baixo (≤6)","Médio (6–8)","Alto (>8)"])
        obs=ds.dropna(subset=["IEG_next"])
        g=(obs.groupby("fx",observed=False)[["queda_IDA","queda_IEG"]].mean()*100).round(1)
        gl=g.reset_index().melt(id_vars="fx",var_name="q",value_name="pct")
        gl["q"]=gl["q"].map({"queda_IDA":"Queda de IDA","queda_IEG":"Queda de IEG"})
        fig=px.bar(gl,x="fx",y="pct",color="q",barmode="group",
                   color_discrete_map={"Queda de IDA":SUB,"Queda de IEG":AMBER},text=fmt_list(gl["pct"],0,"%"))
        fig.update_layout(yaxis_title="% que caiu no ano seguinte",xaxis_title="Faixa de IPS",legend_title="")
        fig.update_traces(textposition="outside")
        st.plotly_chart(style(fig,300),width='stretch')
        insight("O IPS antecede quedas de <b>engajamento</b>: quanto maior o IPS, menor a chance de queda de IEG no ano "
                f"seguinte ({g['queda_IEG'].iloc[0]:.0f}% na faixa baixa vs {g['queda_IEG'].iloc[-1]:.0f}% na alta). "
                "Para o <b>desempenho (IDA)</b>, porém, a relação é fraca — ou seja, o IPS é um sinal precoce de "
                "desengajamento, não diretamente de nota. <i>(Correção metodológica: consideramos apenas alunos com ano "
                "seguinte observável.)</i>")

    # Q6 IPP  /  Q7 IPV
    c1,c2=st.columns(2)
    with c1:
        with st.container(border=True):
            qhead(6,"Aspectos psicopedagógicos (IPP)","Confirmam ou contradizem a defasagem do IAN?")
            def rot(v): return "Em fase" if v>=10 else ("Moderada" if v>=5 else "Severa")
            dd=d.dropna(subset=["IPP","IAN"]).copy(); dd["cat"]=dd["IAN"].apply(rot)
            order=["Em fase","Moderada","Severa"]; m=dd.groupby("cat")["IPP"].mean().reindex(order)
            fig=px.bar(x=order,y=m.values,color=order,color_discrete_map={"Em fase":GOOD,"Moderada":WARN,"Severa":BAD},
                       text=fmt_list(m.values,2))
            fig.update_layout(showlegend=False,yaxis_title="IPP médio",xaxis_title="Nível do IAN",yaxis_range=[0,10])
            fig.update_traces(textposition="outside")
            st.plotly_chart(style(fig,280,legend=False),width='stretch')
            insight(f"Na média o IPP confirma o IAN (em fase {m.get('Em fase',float('nan')):.2f} vs severa "
                    f"{m.get('Severa',float('nan')):.2f}), mas a dispersão individual é grande: use o IPP para olhar o "
                    f"<b>potencial atual</b> do aluno, não o histórico de defasagem.")
    with c2:
        with st.container(border=True):
            qhead(7,"Ponto de virada (IPV)","Quais comportamentos mais influenciam o IPV?")
            cols=[c for c in ["INDE","IDA","IEG","IPP","IAN","IAA","IPS"] if c in d.columns]
            cc=d[["IPV"]+cols].corr()["IPV"].drop("IPV").sort_values()
            fig=px.bar(x=cc.values,y=cc.index,orientation="h",text=fmt_list(cc.values,2),
                       color=cc.values,color_continuous_scale=["#5A6B78",TEAL])
            fig.update_layout(coloraxis_showscale=False,xaxis_title="Correlação com IPV",yaxis_title="")
            fig.update_traces(textposition="outside")
            st.plotly_chart(style(fig,280,legend=False),width='stretch')
            insight("A virada é puxada por <b>desempenho (IDA)</b>, <b>engajamento (IEG)</b> e <b>maturidade "
                    "psicopedagógica (IPP)</b>. Defasagem histórica, autoavaliação e IPS quase não influenciam o salto.")

    # Q8 multidimensional
    with st.container(border=True):
        qhead(8,"Multidimensionalidade","Quais indicadores (IDA+IEG+IPS+IPP) mais elevam a nota global (INDE)?")
        cc=d[["INDE","IDA","IEG","IPS","IPP"]].corr()["INDE"].drop("INDE").sort_values()
        fig=px.bar(x=cc.values,y=cc.index,orientation="h",text=fmt_list(cc.values,2),
                   color=cc.values,color_continuous_scale=["#5A6B78",BLUE])
        fig.update_layout(coloraxis_showscale=False,xaxis_title="Correlação com INDE",yaxis_title="")
        fig.update_traces(textposition="outside")
        st.plotly_chart(style(fig,280,legend=False),width='stretch')
        insight(f"O INDE é alavancado sobretudo pelo binômio <b>IDA</b> (r={cc['IDA']:.2f}) e <b>IEG</b> "
                f"(r={cc['IEG']:.2f}). O IPP entra como suporte (r={cc['IPP']:.2f}) e o IPS como base estável "
                f"(r={cc['IPS']:.2f}). Foco em recuperar engajamento e notas eleva mais a nota global.")

    # Q10 efetividade  /  Q11 insight
    c1,c2=st.columns(2)
    with c1:
        with st.container(border=True):
            qhead(10,"Efetividade do programa","Os indicadores melhoram de forma consistente ao longo do ciclo?")
            t=(pd.crosstab(d["Ano"],d["Faixa_defasagem"],normalize="index")*100)
            tp=(pd.crosstab(d["Ano"],d["Pedra"],normalize="index")*100)
            comp=pd.DataFrame({
                "Métrica":["% em fase","% Topázio"],
                str(a0):[t.loc[a0].get("Em fase",0),tp.loc[a0].get("TOPAZIO",0)],
                str(a1):[t.loc[a1].get("Em fase",0),tp.loc[a1].get("TOPAZIO",0)]})
            cl=comp.melt(id_vars="Métrica",var_name="Ano",value_name="pct")
            fig=px.bar(cl,x="Métrica",y="pct",color="Ano",barmode="group",
                       color_discrete_map={str(a0):SUB,str(a1):TEAL},text=fmt_list(cl["pct"],1,"%"))
            fig.update_layout(yaxis_title="%",xaxis_title="",legend_title="")
            fig.update_traces(textposition="outside")
            st.plotly_chart(style(fig,280),width='stretch')
            insight(f"Do início ao fim do ciclo, <b>% em fase</b> e <b>% Topázio</b> sobem de forma consistente — "
                    f"o programa produz avanço real de trajetória, não só oscilação pontual.")
    with c2:
        with st.container(border=True):
            qhead(11,"Insight extra","A virada alcança quem começou para trás?")
            d0=d[d.Ano==a0][["RA","Faixa_defasagem"]].rename(columns={"Faixa_defasagem":"f0"})
            d1=d[d.Ano==a1][["RA","Faixa_defasagem"]].rename(columns={"Faixa_defasagem":"f1"})
            j=d0.merge(d1,on="RA"); defas=j[j.f0!="Em fase"]
            rec=(defas.f1=="Em fase").mean()*100 if len(defas) else float("nan")
            fig=go.Figure(go.Indicator(mode="gauge+number",value=rec,number={"suffix":"%","font":{"color":GOOD}},
                gauge={"axis":{"range":[0,100]},"bar":{"color":GOOD,"thickness":.32},"bgcolor":"rgba(0,0,0,0)"}))
            fig.update_layout(height=250,margin=dict(t=10,b=0,l=20,r=20),paper_bgcolor="rgba(0,0,0,0)",font=dict(color=TXT))
            st.plotly_chart(fig,width='stretch')
            insight(f"Entre os <b>{len(defas)}</b> alunos que começaram <b>defasados</b> em {a0}, "
                    f"<b>{rec:.0f}%</b> chegaram <b>em fase</b> em {a1}. A defasagem inicial não é destino: "
                    f"a permanência no programa reverte a trajetória.")

# = PREVISAO
else:
    sec("Preditor de risco de defasagem")
    st.markdown("Informe os indicadores atuais de um aluno para estimar a **probabilidade de risco de defasagem no próximo ciclo**.")
    st.caption(f"Modelos treinados apenas com **alunos constantes** ({meta['n_ras_treino']} alunos presentes nos três anos), "
               "que têm trajetória longitudinal real para o alerta precoce.")
    modelo_nome = st.radio("Modelo", list(models.keys()), horizontal=True)
    modelo = models[modelo_nome]

    cL,cR = st.columns([1.05,1])
    with cL:
        st.markdown("##### Indicadores do aluno")
        defas=st.slider("Defasagem atual (anos) · 0 = na série certa",-5,3,0,1)
        c1,c2=st.columns(2)
        with c1:
            ipp=st.slider("IPP · Psicopedagógico",0.0,10.0,6.0,0.1)
            ipv=st.slider("IPV · Ponto de Virada",0.0,10.0,6.0,0.1)
            ips=st.slider("IPS · Psicossocial",0.0,10.0,6.5,0.1)
        with c2:
            ida=st.slider("IDA · Desempenho",0.0,10.0,6.0,0.1)
            ieg=st.slider("IEG · Engajamento",0.0,10.0,7.0,0.1)
            iaa=st.slider("IAA · Autoavaliação",0.0,10.0,8.0,0.1)
        vals={"Defasagem":defas,"IPP":ipp,"IPV":ipv,"IDA":ida,"IPS":ips,"IEG":ieg,"IAA":iaa}
        prob=core.predict_risk(modelo,vals)
        prob_outro={n:core.predict_risk(m,vals) for n,m in models.items()}
        banda,_=core.risk_band(prob)
        cor=BAD if prob>=0.66 else (WARN if prob>=0.40 else GOOD)
    with cR:
        st.markdown("##### Resultado")
        g=go.Figure(go.Indicator(mode="gauge+number",value=prob*100,
            number={"suffix":"%","font":{"size":46,"color":cor}},
            gauge={"axis":{"range":[0,100],"tickcolor":SUB},"bar":{"color":cor,"thickness":.32},
                   "bgcolor":"rgba(0,0,0,0)",
                   "steps":[{"range":[0,40],"color":"rgba(53,192,138,.18)"},
                            {"range":[40,66],"color":"rgba(242,177,74,.18)"},
                            {"range":[66,100],"color":"rgba(232,106,92,.18)"}]}))
        g.update_layout(height=250,margin=dict(t=18,b=0,l=20,r=20),paper_bgcolor="rgba(0,0,0,0)",font=dict(color=TXT))
        st.plotly_chart(g, width='stretch')
        st.markdown(f"<div style='text-align:center;font-size:22px;font-weight:800;color:{cor}'>{banda} · {modelo_nome}</div>",
                    unsafe_allow_html=True)
        outro=[f"{n}: {p*100:.1f}%" for n,p in prob_outro.items()]
        st.caption("Comparação · " + "  |  ".join(outro))
        if prob>=0.66: st.error("Prioridade de acompanhamento: sugere-se plano psicopedagógico individualizado.")
        elif prob>=0.40: st.warning("Atenção: monitorar engajamento e desempenho no próximo ciclo.")
        else: st.success("Situação favorável: manter o acompanhamento regular.")

    st.markdown("---")
    d1,d2=st.columns(2)
    with d1:
        sec("Comparação dos modelos (ROC)")
        with st.container(border=True):
            figroc=go.Figure()
            fr,tr=meta["roc_rf"]; fn,tn=meta["roc_nn"]
            figroc.add_trace(go.Scatter(x=fr,y=tr,mode="lines",name=f"Random Forest (AUC={meta['auc_rf']:.2f})",line=dict(color=TEAL,width=3)))
            figroc.add_trace(go.Scatter(x=fn,y=tn,mode="lines",name=f"Rede Neural (AUC={meta['auc_nn']:.2f})",line=dict(color=AMBER,width=3)))
            figroc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(color=SUB,dash="dash",width=1),showlegend=False))
            figroc.update_layout(xaxis_title="Falso positivo",yaxis_title="Verdadeiro positivo",legend=dict(x=.35,y=.05))
            st.plotly_chart(style(figroc,320), width='stretch')
    with d2:
        sec("O que mais pesa (Random Forest)")
        with st.container(border=True):
            imp=meta["importances"].sort_values()
            figi=px.bar(x=imp.values,y=[core.FEATURE_LABELS[i] for i in imp.index],orientation="h",
                        text=fmt_list(imp.values,3),color_discrete_sequence=[TEAL])
            figi.update_traces(textposition="outside")
            figi.update_layout(xaxis_title="Importância",yaxis_title="")
            st.plotly_chart(style(figi,320,legend=False), width='stretch')

    with st.container(border=True):
        st.markdown(f"""##### Sobre os modelos
- **Objetivo:** prever se o aluno estará defasado no **ano seguinte** (alerta precoce).
- **Amostra:** treinados **apenas com alunos constantes** — {meta['n_ras_treino']} alunos presentes nos três anos ({meta['n']:,} pares aluno-ano · {meta['base_rate']:.0f}% em risco na base).
- **Random Forest** · AUC **{meta['auc_rf']:.2f}**  ·  **Rede Neural (MLP 16-8)** · AUC **{meta['auc_nn']:.2f}**.
- **Normalização:** a Rede Neural usa `StandardScaler`, colocando os indicadores na mesma escala, de modo que **Defasagem** (−5 a 3) não enviese frente aos indicadores de 0 a 10.
""".replace(",","."))
        st.caption("Ferramenta de apoio à decisão — não substitui a avaliação da equipe pedagógica.")

    st.markdown("---")
    sec("Avaliar vários alunos (opcional)")
    st.caption("Envie um CSV com as colunas: " + ", ".join(core.FEATURES))
    up=st.file_uploader("Upload de CSV",type=["csv"],label_visibility="collapsed")
    if up is not None:
        try:
            b=pd.read_csv(up); falta=[c for c in core.FEATURES if c not in b.columns]
            if falta: st.error("Faltam colunas: "+", ".join(falta))
            else:
                b=b.copy(); b["prob_risco"]=modelo.predict_proba(b[core.FEATURES])[:,1].round(3)
                b["faixa"]=b["prob_risco"].apply(lambda p:core.risk_band(p)[0])
                st.dataframe(b.sort_values("prob_risco",ascending=False), width='stretch')
                st.download_button("Baixar resultados", b.to_csv(index=False).encode("utf-8"),
                                   "risco_alunos.csv","text/csv")
        except Exception as e:
            st.error(f"Não foi possível processar: {e}")

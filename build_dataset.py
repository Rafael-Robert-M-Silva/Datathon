import pandas as pd, numpy as np, re, unicodedata
from datetime import datetime
X="/mnt/user-data/uploads/BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx"
d22=pd.read_excel(X,"PEDE2022"); d23=pd.read_excel(X,"PEDE2023"); d24=pd.read_excel(X,"PEDE2024")
d22["Ano"]=2022; d23["Ano"]=2023; d24["Ano"]=2024
d22=d22.rename(columns={"Idade 22":"Idade","Pedra 22":"Pedra","INDE 22":"INDE","Matem":"Mat","Portug":"Por","Inglês":"Ing","Fase ideal":"Fase Ideal","Defas":"Defasagem"})
for c in ["INDE","IAN","IDA","IEG","IAA","IPS","IPV"]: d22[c]=pd.to_numeric(d22[c],errors="coerce")
d22["IPP"]=np.where(d22["Fase"]<=7,(d22["INDE"]-(d22["IAN"]*0.1+d22["IDA"]*0.2+d22["IEG"]*0.2+d22["IAA"]*0.1+d22["IPS"]*0.1+d22["IPV"]*0.2))/0.1,np.nan)
std=['Ano','RA','Fase','Turma','Idade','Gênero','Ano ingresso','Instituição de ensino','Pedra','INDE','Nº Av','IAA','IEG','IPS','IPP','IDA','Mat','Por','Ing','IPV','IAN','Fase Ideal','Defasagem']
d22=d22[std]; d22["Gênero"]=d22["Gênero"].replace({"Menino":"Masculino","Menina":"Feminino"})
d23=d23.rename(columns={"Pedra 2023":"Pedra","INDE 2023":"INDE"})[std]
d24=d24.rename(columns={"Pedra 2024":"Pedra","INDE 2024":"INDE"})[std]
df=pd.concat([d22,d23,d24],ignore_index=True)

def extrair_fase(v):
    if pd.isna(v): return pd.NA
    if isinstance(v,(int,float)): return int(v)
    v=str(v).strip().upper()
    if v.startswith("ALFA"): return 0
    m=re.search(r"\d+",v); return int(m.group()) if m else pd.NA
df["Fase_num"]=df["Fase"].apply(extrair_fase)
def tratar_idade(v):
    if pd.isna(v): return pd.NA
    if isinstance(v,(int,float)): return int(v)
    if isinstance(v,datetime): return v.day
    return pd.NA
df["Idade"]=df["Idade"].apply(tratar_idade).astype("Int64")
def padroniza(t):
    if pd.isna(t): return pd.NA
    t=str(t).strip(); t=''.join(c for c in unicodedata.normalize('NFKD',t) if not unicodedata.combining(c)).upper()
    return pd.NA if t=="INCLUIR" else t
df["Pedra"]=df["Pedra"].apply(padroniza)
for c in ["INDE","IAA","IEG","IPS","IPP","IDA","IPV","IAN","Defasagem","Mat","Por","Ing"]:
    df[c]=pd.to_numeric(df[c],errors="coerce")
cond=[df["Defasagem"]>=0,(df["Defasagem"]<0)&(df["Defasagem"]>=-2),df["Defasagem"]<-2]
df["Faixa_defasagem"]=np.select(cond,["Em fase","Moderada","Severa"],default=None)
nome={0:"Alfa",1:"Fase 1",2:"Fase 2",3:"Fase 3",4:"Fase 4",5:"Fase 5",6:"Fase 6",7:"Fase 7",8:"Fase 8"}
df["Fase_label"]=df["Fase_num"].map(nome)

cols=['Ano','RA','Ano ingresso','Fase_num','Fase_label','Idade','Gênero','Instituição de ensino','Pedra','INDE',
      'IAA','IEG','IPS','IPP','IDA','Mat','Por','Ing','IPV','IAN','Defasagem','Faixa_defasagem']
out=df[cols].copy()
out.to_csv("/home/claude/dados_consolidados.csv",index=False,encoding="utf-8")
print("linhas:",len(out),"colunas:",len(out.columns))
print(out.head(2).to_string())

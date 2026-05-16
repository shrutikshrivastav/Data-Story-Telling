import os, io
import pandas as pd, numpy as np
import plotly.express as px, plotly.figure_factory as ff
from flask import Flask, render_template, request, jsonify, send_file
from datetime import datetime

app = Flask(__name__)

def clean_data(df):
    before_shape = df.shape
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    dup = df.duplicated().sum(); df = df.drop_duplicates()
    miss = df.isnull().sum().sum()
    df = df.dropna(how="all").fillna(method="ffill").fillna(method="bfill")
    num = df.select_dtypes(include=np.number).columns.tolist()
    cat = df.select_dtypes(exclude=np.number).columns.tolist()
    for c in cat:
        try: df[c] = pd.to_datetime(df[c])
        except: pass
    out = detect_outliers(df[num]) if num else 0
    after_shape = df.shape
    score = max(0, 100 - (dup + miss + out))
    summary = {"before_rows":before_shape[0],"before_cols":before_shape[1],
               "after_rows":after_shape[0],"after_cols":after_shape[1],
               "duplicates_removed":int(dup),"missing_values_fixed":int(miss),
               "outliers_detected":int(out),"quality_score":score}
    return df, summary, num, cat

def detect_outliers(df):
    out=0
    for c in df.columns:
        q1,q3=df[c].quantile(0.25),df[c].quantile(0.75)
        iqr=q3-q1; low,up=q1-1.5*iqr,q3+1.5*iqr
        out+=((df[c]<low)|(df[c]>up)).sum()
    return int(out)

def generate_visualizations(df,num,cat):
    figs=[]
    if num:
        figs.append(px.histogram(df,x=num[0],title="Distribution Histogram"))
        figs.append(px.scatter(df,x=num[0],y=num[-1],title="Scatter Plot"))
        figs.append(px.line(df,x=df.index,y=num[0],title="Trend Line"))
        corr=df[num].corr()
        heat=ff.create_annotated_heatmap(z=corr.values,x=list(corr.columns),y=list(corr.index),colorscale="Viridis")
        heat.update_layout(title="Correlation Heatmap"); figs.append(heat)
    if cat:
        figs.append(px.bar(df,x=cat[0],title="Category Bar Chart"))
        figs.append(px.pie(df,names=cat[0],title="Category Pie Chart"))
    for f in figs: f.update_layout(template="plotly_dark",transition_duration=500)
    return [f.to_html(full_html=False) for f in figs]

def generate_narrative(df,s,num,cat):
    ins=[f"Dataset cleaned: {s['after_rows']} rows, {s['after_cols']} cols.",
         f"Duplicates removed {s['duplicates_removed']}, Missing fixed {s['missing_values_fixed']}, Outliers {s['outliers_detected']}.",
         f"Quality score {s['quality_score']}%."] 
    if num:
        for c in num: ins.append(f"'{c}' range {df[c].min():.2f}-{df[c].max():.2f}, avg {df[c].mean():.2f}.")
    if cat:
        for c in cat: ins.append(f"'{c}' dominated by '{df[c].value_counts().idxmax()}'.")
    ins.append("Report generated "+datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return ins

@app.route("/")
def index(): return render_template("index.html")

@app.route("/upload",methods=["POST"])
def upload():
    f=request.files["file"]
    df=pd.read_csv(f) if f.filename.endswith(".csv") else pd.read_excel(f)
    cleaned,s,num,cat=clean_data(df)
    return jsonify({"summary":s,"visualizations":generate_visualizations(cleaned,num,cat),"narrative":generate_narrative(cleaned,s,num,cat)})

@app.route("/download",methods=["POST"])
def download():
    f=request.files["file"]
    df=pd.read_csv(f) if f.filename.endswith(".csv") else pd.read_excel(f)
    cleaned,_,_,_=clean_data(df)
    buf=io.BytesIO(); cleaned.to_csv(buf,index=False); buf.seek(0)
    return send_file(buf,mimetype="text/csv",as_attachment=True,download_name=f"cleaned_{f.filename}")

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)))

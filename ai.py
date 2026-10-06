"""ai.py — Generative AI (Google Gemini): สรุปผล วิเคราะห์สาเหตุ และให้ข้อเสนอแนะเชิงธุรกิจ
หลักการ: ตัวเลขทั้งหมดคำนวณด้วยโค้ด Python แล้วส่งให้ AI เล่าเป็นภาษาคน (AI ไม่คำนวณเอง จึงไม่แต่งตัวเลข)"""
import os
import time

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
DOW = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]


def _f(v, d=0, suf=""):
    return "N/A" if v is None or pd.isna(v) else f"{v:,.{d}f}{suf}"


def build_context(k, products, daily, reg, ts_fc, backtest) -> str:
    dow = daily.groupby(daily["Date"].dt.dayofweek)["Sales"].mean()
    L = [f"ช่วงข้อมูล: {k['start']} ถึง {k['end']} ({k['days']} วัน)",
         f"ยอดขายรวม {_f(k['total'])} | เฉลี่ย/วัน {_f(k['avg_daily'])}",
         f"30 วันล่าสุด {_f(k['last30'])} (เทียบ 30 วันก่อนหน้า {_f(k['growth30'], 1, '%')})",
         f"วันขายดีสุด {DOW[int(dow.idxmax())]} ({_f(dow.max())}), ต่ำสุด {DOW[int(dow.idxmin())]} ({_f(dow.min())})"]
    if pd.notna(k["uplift"]):
        L.append(f"โปรโมชัน: เฉลี่ย {_f(k['promo_avg'])} vs วันปกติ {_f(k['normal_avg'])} ({_f(k['uplift'], 1, '%')})")
    L.append("สินค้า (ยอดขาย | ส่วนแบ่ง | ABC | โต 30 วัน | กลุ่ม):")
    for _, r in products.iterrows():
        L.append(f"- {r.Product} ({r.Category}): {_f(r.Sales)} | {_f(r.Share, 1, '%')} | {r.ABC} | {_f(r.Growth30d, 1, '%')} | {r.Action}")
    if reg:
        b = reg["metrics"].set_index("Model").loc[reg["best"]]
        top = reg["importance"].sort_values(ascending=False).head(3)
        L.append(f"Regression ({reg['best']}): MAE {_f(b.MAE)}, RMSE {_f(b.RMSE)}, R2 {b.R2:.3f}, baseline MAE {_f(reg['naive_mae'])}")
        L.append("ปัจจัยสำคัญ: " + ", ".join(f"{i} ({v:.0%})" for i, v in top.items()))
    if ts_fc is not None:
        L.append(f"พยากรณ์ Time Series {len(ts_fc)} วัน: รวม {_f(ts_fc.Forecast.sum())}, เฉลี่ย/วัน {_f(ts_fc.Forecast.mean())}")
    if backtest:
        L.append(f"Backtest 28 วัน: MAE Holt-Winters {_f(backtest['mae'])} vs baseline {_f(backtest['naive_mae'])}")
    return "\n".join(L)


def ask_ai(question: str, context: str) -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("ยังไม่ได้ตั้งค่า GEMINI_API_KEY ในไฟล์ .env")
    prompt = f"""ข้อมูลสรุปยอดขายของร้านค้า:
{context}

คำถาม: {question}

ตอบเป็นภาษาไทย อ้างอิงตัวเลขที่ให้เท่านั้น จัดโครงสร้างเป็น
## สรุปผล
## สาเหตุที่เป็นไปได้ (ระบุว่าเป็นข้อสันนิษฐานจากข้อมูล)
## ข้อเสนอแนะเชิงธุรกิจที่ทำได้จริง
ห้ามสร้างตัวเลข ข่าว หรือปัจจัยภายนอกที่ไม่มีในข้อมูล และไม่รับประกันยอดขายในอนาคต"""
    client, err = genai.Client(api_key=key), None
    for wait in (0, 2, 5):  # ลองใหม่เมื่อเซิร์ฟเวอร์ไม่ว่าง
        time.sleep(wait)
        try:
            r = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3,
                        system_instruction="คุณคือนักวิเคราะห์ค้าปลีกที่ให้คำแนะนำเชิงปฏิบัติ กระชับ ตรงประเด็น"))
            return r.text
        except Exception as e:
            err = e
            if not any(c in str(e) for c in ("503", "429", "UNAVAILABLE")):
                raise
    raise err

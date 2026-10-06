# RetailMind AI — เว็บ AI วิเคราะห์ร้านค้า (Python + Streamlit)

## โครงสร้าง
| ไฟล์ | หน้าที่ |
|---|---|
| `data.py` | สร้างข้อมูลจำลอง, โหลด/ทำความสะอาด CSV-Excel, KPI, Pareto/ABC, การเติบโตสินค้า |
| `models.py` | Regression (Ridge, Random Forest), Permutation Importance, Time Series (Holt-Winters), Backtest |
| `ai.py` | Generative AI — Google Gemini API (สรุปผล สาเหตุ ข้อเสนอแนะ) |
| `app.py` | Dashboard (Streamlit + Plotly) และพื้นหลังขยับได้ด้วย CSS |

## วิธีรัน
```
pip install -r requirements.txt
cp .env.example .env      # แล้วใส่ GEMINI_API_KEY (ขอฟรีที่ https://aistudio.google.com/apikey)
streamlit run app.py
```
ไม่ใส่ key ก็ใช้ Dashboard / Regression / Forecast ได้ครบ ยกเว้นปุ่ม Gen-AI

## API Key
ใช้ **Google Gemini API** (ไลบรารี `google-genai`) ตั้งค่าใน `.env`: `GEMINI_API_KEY`, `GEMINI_MODEL`
**ห้ามอัปโหลดไฟล์ `.env` ขึ้น GitHub หรือส่งให้ผู้อื่น**

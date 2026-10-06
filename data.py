"""data.py — สร้าง/โหลด/ทำความสะอาดข้อมูล และคำนวณสถิติยอดขาย (ไม่มี ML ในไฟล์นี้)"""
import numpy as np
import pandas as pd

# (สินค้า, หมวด, ราคา, แนวโน้มการเติบโตตลอด 2 ปี)
CATALOG = [
    ("Latte", "เครื่องดื่ม", 65, 0.2), ("Americano", "เครื่องดื่ม", 55, 0.0),
    ("Green Tea", "เครื่องดื่ม", 50, 0.9), ("Croissant", "เบเกอรี่", 70, -0.1),
    ("Cheesecake", "เบเกอรี่", 95, 0.3), ("Sandwich", "อาหาร", 89, 0.1),
    ("Pasta", "อาหาร", 129, -0.45), ("Coffee Beans", "สินค้าปลีก", 350, 0.5),
]

ALIASES = {
    "Date": ["date", "วันที่", "order date", "transaction date"],
    "Product": ["product", "item", "product name", "สินค้า", "ชื่อสินค้า"],
    "Quantity": ["quantity", "qty", "units", "จำนวน"],
    "Price": ["price", "unit price", "ราคา"],
    "Category": ["category", "หมวดหมู่"],
    "Promotion": ["promotion", "promo", "โปรโมชัน"],
}


def generate_sample(days: int = 730) -> pd.DataFrame:
    """ข้อมูลจำลองร้านกาแฟ/เบเกอรี่ย้อนหลัง 2 ปี (เมล็ดสุ่มคงที่ ผลซ้ำได้)"""
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)
    rows = []
    for i, d in enumerate(dates):
        weekend = d.weekday() >= 5
        promo = int(rng.random() < 0.15)
        base = (1 + 0.3 * i / days) * (1.35 if weekend else 1) * (1.18 if promo else 1)
        base *= 1 + 0.1 * np.sin(2 * np.pi * i / 365)
        for k, (name, cat, price, growth) in enumerate(CATALOG):
            if rng.random() < 0.85:
                trend = max(0.25, 1 + growth * i / days)
                qty = max(1, round((12 if weekend else 8) * base * trend * rng.uniform(0.6, 1.4) * (0.3 if k == 7 else 1)))
                rows.append((d, name, cat, qty, price, qty * price, promo))
    return pd.DataFrame(rows, columns=["Date", "Product", "Category", "Quantity", "Price", "Sales", "Promotion"])


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    """จับคู่ชื่อคอลัมน์ที่หลากหลาย -> ชื่อมาตรฐาน และตรวจความถูกต้อง"""
    df = df.copy()
    rename = {}
    for col in df.columns:
        for std, names in ALIASES.items():
            if str(col).strip().lower() in names:
                rename[col] = std
    df = df.rename(columns=rename)

    missing = [c for c in ("Product", "Quantity", "Price") if c not in df.columns]
    if missing:
        raise ValueError("ไม่พบคอลัมน์ที่จำเป็น: " + ", ".join(missing))

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["Date"])
    else:
        df["Date"] = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(df))
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(1)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)
    df["Sales"] = df["Quantity"] * df["Price"]
    df["Category"] = df["Category"].fillna("General") if "Category" in df else "General"
    df["Promotion"] = pd.to_numeric(df["Promotion"], errors="coerce").fillna(0).clip(0, 1) if "Promotion" in df else 0
    if len(df) < 30:
        raise ValueError("ข้อมูลน้อยเกินไป (ต้องมีอย่างน้อย 30 แถว)")
    return df


def daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """รวมเป็นรายวัน และเติมวันที่ขาดหายด้วยค่าวันก่อนหน้า"""
    d = df.groupby("Date").agg(Sales=("Sales", "sum"), Quantity=("Quantity", "sum"),
                               Promotion=("Promotion", "max"), AvgPrice=("Price", "mean"))
    d = d.reindex(pd.date_range(d.index.min(), d.index.max())).ffill()
    d.index.name = "Date"
    return d.reset_index()


def overall_kpis(daily: pd.DataFrame) -> dict:
    last30, prev30 = daily["Sales"].tail(30).sum(), daily["Sales"].iloc[-60:-30].sum()
    promo = daily.groupby("Promotion")["Sales"].mean()
    uplift = (promo[1] / promo[0] - 1) * 100 if {0, 1} <= set(promo.index) and promo[0] > 0 else np.nan
    return {
        "start": daily["Date"].min().date(), "end": daily["Date"].max().date(), "days": len(daily),
        "total": daily["Sales"].sum(), "avg_daily": daily["Sales"].mean(), "qty": daily["Quantity"].sum(),
        "last30": last30, "growth30": (last30 - prev30) / prev30 * 100 if prev30 > 0 else np.nan,
        "promo_avg": promo.get(1, np.nan), "normal_avg": promo.get(0, np.nan), "uplift": uplift,
    }


def product_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Pareto/ABC + การเติบโต 30 วัน + คำแนะนำเชิงกลยุทธ์ต่อสินค้า"""
    p = (df.groupby(["Product", "Category"], as_index=False)
           .agg(Sales=("Sales", "sum"), Quantity=("Quantity", "sum"))
           .sort_values("Sales", ascending=False).reset_index(drop=True))
    p["Share"] = p["Sales"] / p["Sales"].sum() * 100
    p["CumShare"] = p["Share"].cumsum()
    before = p["CumShare"] - p["Share"]
    p["ABC"] = np.where(before < 70, "A", np.where(before < 90, "B", "C"))

    end = df["Date"].max()
    now = df[df["Date"] > end - pd.Timedelta(days=30)].groupby("Product")["Sales"].sum()
    prev = df[(df["Date"] <= end - pd.Timedelta(days=30)) & (df["Date"] > end - pd.Timedelta(days=60))].groupby("Product")["Sales"].sum()
    p["Growth30d"] = [((now.get(x, 0) - prev.get(x, 0)) / prev[x] * 100) if prev.get(x, 0) > 0 else np.nan for x in p["Product"]]

    med, g = p["Share"].median(), p["Growth30d"].fillna(0)
    p["Action"] = np.select(
        [(p["Share"] >= med) & (g >= 0), (p["Share"] < med) & (g > 0), p["Share"] >= med],
        ["⭐ Star — ลงทุนต่อ", "🚀 Rising — เร่งโปรโมต", "⚠️ Defend — แก้ยอดตก"], "🔍 Review — ทบทวน")
    return p

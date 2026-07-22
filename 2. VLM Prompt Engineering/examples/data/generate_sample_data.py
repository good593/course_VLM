"""
실습용 샘플 이미지를 전부 코드로 생성한다 (외부 다운로드/저작권 이슈 없음).
한 번만 실행하면 되고, 노트북들은 여기서 생성된 data/*.png 를 그대로 사용한다.

    uv run python data/generate_sample_data.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import koreanize_matplotlib  # noqa: F401  (import만으로 한글 폰트 적용)
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = Path(__file__).parent
FONT_DIR = Path("C:/Windows/Fonts")
FONT_KR = FONT_DIR / "malgun.ttf"
FONT_KR_BOLD = FONT_DIR / "malgunbd.ttf"


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


# ---------------------------------------------------------------------------
# 1. 상품 이미지 (이미지 정보 추출 실습용)
#    정답: 카테고리=상의(티셔츠), 색상=네이비, 패턴=스트라이프, 사이즈=M
# ---------------------------------------------------------------------------
def make_product_image():
    img = Image.new("RGB", (600, 600), "white")
    d = ImageDraw.Draw(img)

    navy = (30, 50, 90)
    white = (255, 255, 255)

    # 티셔츠 몸통
    body = [(180, 180), (420, 180), (420, 480), (180, 480)]
    d.polygon(body, fill=navy)
    # 소매
    d.polygon([(180, 180), (110, 230), (150, 300), (200, 250)], fill=navy)
    d.polygon([(420, 180), (490, 230), (450, 300), (400, 250)], fill=navy)
    # 목 (칼라)
    d.ellipse([(265, 165), (335, 205)], fill=white)
    d.pieslice([(265, 165), (335, 205)], 0, 180, fill=navy)

    # 스트라이프 패턴
    for y in range(200, 470, 40):
        d.rectangle([182, y, 418, y + 15], fill=white)

    # 사이즈 태그
    d.rectangle([440, 200, 560, 240], fill=(240, 240, 240), outline=(120, 120, 120))
    d.text((452, 210), "SIZE: M", font=font(FONT_KR_BOLD, 18), fill=(40, 40, 40))

    img.save(DATA_DIR / "product.png")


# ---------------------------------------------------------------------------
# 2. 영수증 (Document AI / OCR vs VLM 실습용)
#    정답: 상호=행복마트, 날짜=2026-07-20, 합계=5,300원
# ---------------------------------------------------------------------------
def make_receipt_image():
    img = Image.new("RGB", (420, 520), "white")
    d = ImageDraw.Draw(img)
    f_title = font(FONT_KR_BOLD, 26)
    f_body = font(FONT_KR, 17)
    f_small = font(FONT_KR, 14)

    def hr(y):
        d.line([(30, y), (390, y)], fill="black", width=1)

    def centered(text, fnt, y):
        bbox = d.textbbox((0, 0), text, font=fnt)
        w = bbox[2] - bbox[0]
        d.text(((img.width - w) // 2, y), text, font=fnt, fill="black")

    def row(name, price, y, fnt=f_body):
        d.text((30, y), name, font=fnt, fill="black")
        bbox = d.textbbox((0, 0), price, font=fnt)
        w = bbox[2] - bbox[0]
        d.text((390 - w, y), price, font=fnt, fill="black")

    y = 30
    centered("행복마트", f_title, y)
    y += 45
    centered("서울시 강남구 테헤란로 123", f_small, y)
    y += 28
    centered("2026-07-20 14:32", f_small, y)
    y += 40
    hr(y)
    y += 20
    row("생수 500ml x2", "2,000", y)
    y += 34
    row("삼각김밥 x1", "1,500", y)
    y += 34
    row("초코우유 x1", "1,800", y)
    y += 30
    hr(y)
    y += 20
    row("합계", "5,300원", y, fnt=f_title)
    y += 46
    hr(y)
    y += 24
    centered("이용해 주셔서 감사합니다", f_small, y)

    img.save(DATA_DIR / "receipt.png")


# ---------------------------------------------------------------------------
# 3. 계약서 일부 (Document AI 실습용)
#    정답: 계약기간=2026-08-01~2028-07-31, 위약금=보증금의 10%
# ---------------------------------------------------------------------------
def make_contract_image():
    img = Image.new("RGB", (700, 500), "white")
    d = ImageDraw.Draw(img)
    f_title = font(FONT_KR_BOLD, 24)
    f_body = font(FONT_KR, 16)

    d.text((260, 30), "임대차 계약서", font=f_title, fill="black")

    clauses = [
        "제1조 (계약기간)",
        "  본 계약의 임대차 기간은 2026년 8월 1일부터",
        "  2028년 7월 31일까지로 한다.",
        "",
        "제2조 (위약금)",
        "  임차인이 계약을 조기 해지할 경우, 보증금의 10%를",
        "  위약금으로 지급한다.",
        "",
        "제3조 (관리비)",
        "  관리비는 월 150,000원으로 별도 부과한다.",
    ]
    y = 100
    for line in clauses:
        d.text((60, y), line, font=f_body, fill="black")
        y += 34

    img.save(DATA_DIR / "contract.png")
    img.convert("RGB").save(DATA_DIR / "contract.pdf")  # Document AI - PDF 이해 실습용


# ---------------------------------------------------------------------------
# 4. 표 이미지 (Document AI - 표 분석 실습용)
#    정답: 매출 최대 연도 = 2024년 (320억원)
# ---------------------------------------------------------------------------
def make_table_image():
    years = ["2021년", "2022년", "2023년", "2024년"]
    revenue = [180, 210, 260, 320]
    profit = [12, 18, 25, 41]

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=[[y, f"{r}", f"{p}"] for y, r, p in zip(years, revenue, profit)],
        colLabels=["연도", "매출(억원)", "영업이익(억원)"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    fig.suptitle("연도별 매출/영업이익", fontsize=13)
    fig.tight_layout()
    fig.savefig(DATA_DIR / "table.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. 막대그래프 (차트 분석 실습용)
#    정답: 최댓값=5월(820), 최솟값=2월(310)
# ---------------------------------------------------------------------------
def make_bar_chart():
    months = ["1월", "2월", "3월", "4월", "5월", "6월"]
    sales = [450, 310, 520, 610, 820, 590]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(months, sales, color="#4C72B0")
    ax.set_title("2026년 상반기 월별 매출 (백만원)")
    ax.set_ylabel("매출 (백만원)")
    for b, v in zip(bars, sales):
        ax.text(b.get_x() + b.get_width() / 2, v + 10, str(v), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(DATA_DIR / "bar_chart.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. 선그래프 (차트 분석 실습용)
#    정답: 7일차 이후 급격히 하락 (서버 장애 구간 가정)
# ---------------------------------------------------------------------------
def make_line_chart():
    days = list(range(1, 15))
    visitors = [1200, 1250, 1300, 1280, 1350, 1400, 1420, 600, 650, 900, 1100, 1300, 1380, 1450]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(days, visitors, marker="o", color="#DD8452")
    ax.set_title("최근 14일 일별 방문자 수")
    ax.set_xlabel("일차")
    ax.set_ylabel("방문자 수")
    ax.axvspan(7.5, 8.5, color="red", alpha=0.15)
    fig.tight_layout()
    fig.savefig(DATA_DIR / "line_chart.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. 대시보드 (여러 차트 + KPI 조합, 차트 분석 실습용)
# ---------------------------------------------------------------------------
def make_dashboard():
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))

    months = ["3월", "4월", "5월", "6월"]
    this_month = [520, 610, 820, 590]
    last_month = [480, 560, 700, 610]
    axes[0, 0].bar(
        [i - 0.2 for i in range(4)], last_month, width=0.4, label="작년", color="#8C8C8C"
    )
    axes[0, 0].bar(
        [i + 0.2 for i in range(4)], this_month, width=0.4, label="올해", color="#4C72B0"
    )
    axes[0, 0].set_xticks(range(4))
    axes[0, 0].set_xticklabels(months)
    axes[0, 0].set_title("월별 매출 비교")
    axes[0, 0].legend(fontsize=8)

    days = list(range(1, 11))
    trend = [100, 110, 105, 130, 150, 170, 160, 190, 210, 230]
    axes[0, 1].plot(days, trend, marker="o", color="#DD8452")
    axes[0, 1].set_title("가입자 수 추이")

    labels = ["신규", "재구매", "이탈"]
    sizes = [45, 40, 15]
    axes[1, 0].pie(sizes, labels=labels, autopct="%1.0f%%", colors=["#4C72B0", "#55A868", "#C44E52"])
    axes[1, 0].set_title("고객 구성 비율")

    axes[1, 1].axis("off")
    axes[1, 1].text(0.5, 0.6, "820", fontsize=48, ha="center", fontweight="bold", color="#4C72B0")
    axes[1, 1].text(0.5, 0.35, "이번 달 매출 (백만원)", fontsize=13, ha="center")
    axes[1, 1].text(0.5, 0.15, "전월 대비 +17%", fontsize=12, ha="center", color="#55A868")

    fig.suptitle("영업 대시보드 - 2026년 6월", fontsize=15)
    fig.tight_layout()
    fig.savefig(DATA_DIR / "dashboard.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. 웹페이지 와이어프레임 (UI 이해 실습용)
# ---------------------------------------------------------------------------
def make_webpage_mock():
    img = Image.new("RGB", (900, 700), "white")
    d = ImageDraw.Draw(img)

    f_title = font(FONT_KR_BOLD, 26)
    f_nav = font(FONT_KR, 16)
    f_body = font(FONT_KR, 14)

    # 헤더
    d.rectangle([0, 0, 900, 70], fill=(20, 20, 30))
    d.rectangle([30, 20, 90, 50], fill=(255, 255, 255))  # 로고 박스
    for i, item in enumerate(["기능", "요금제", "고객지원"]):
        d.text((550 + i * 100, 25), item, font=f_nav, fill="white")

    # 히어로 섹션
    d.rectangle([0, 70, 900, 320], fill=(245, 247, 250))
    d.text((300, 140), "클라우드로 더 빠르게 시작하세요", font=f_title, fill=(20, 20, 30))
    d.text((320, 190), "가입 후 바로 사용 가능한 무료 플랜 제공", font=f_body, fill=(90, 90, 90))
    d.rounded_rectangle([370, 240, 530, 285], radius=8, fill=(70, 110, 230))
    d.text((400, 253), "지금 시작하기", font=f_nav, fill="white")

    # 기능 카드 3개
    for i in range(3):
        x = 60 + i * 280
        d.rectangle([x, 360, x + 240, 560], outline=(210, 210, 210), width=2)
        d.rectangle([x + 20, 380, x + 80, 440], fill=(70, 110, 230))
        d.text((x + 20, 460), f"핵심 기능 {i + 1}", font=f_nav, fill=(20, 20, 30))
        d.text((x + 20, 495), "간단한 설명 텍스트가", font=f_body, fill=(120, 120, 120))
        d.text((x + 20, 515), "여기에 표시됩니다", font=f_body, fill=(120, 120, 120))

    # 푸터
    d.rectangle([0, 620, 900, 700], fill=(20, 20, 30))
    d.text((30, 645), "© 2026 Course VLM Inc.", font=f_body, fill=(200, 200, 200))

    img.save(DATA_DIR / "webpage.png")


# ---------------------------------------------------------------------------
# 9. 모바일 앱 화면 (Figma 목업, UI 이해 실습용)
# ---------------------------------------------------------------------------
def make_figma_mock():
    img = Image.new("RGB", (420, 780), (250, 250, 250))
    d = ImageDraw.Draw(img)

    f_title = font(FONT_KR_BOLD, 22)
    f_body = font(FONT_KR, 15)
    f_small = font(FONT_KR, 12)

    # 상태바
    d.rectangle([0, 0, 420, 30], fill=(255, 255, 255))
    d.text((370, 6), "9:41", font=f_small, fill="black")

    # 타이틀 + 프로필
    d.text((25, 45), "마이페이지", font=f_title, fill=(20, 20, 30))
    d.ellipse([320, 40, 380, 100], fill=(200, 210, 240))

    # 리스트 항목 4개
    items = ["주문 내역", "배송지 관리", "쿠폰함", "설정"]
    y = 140
    for item in items:
        d.rectangle([25, y, 395, y + 70], fill="white", outline=(225, 225, 225))
        d.rectangle([45, y + 20, 75, y + 50], fill=(70, 110, 230))  # 아이콘 자리
        d.text((95, y + 25), item, font=f_body, fill=(30, 30, 30))
        d.text((355, y + 25), ">", font=f_body, fill=(150, 150, 150))
        y += 90

    # 하단 내비게이션 (4개 아이콘)
    d.rectangle([0, 710, 420, 780], fill="white", outline=(225, 225, 225))
    nav_labels = ["홈", "검색", "장바구니", "마이"]
    for i, label in enumerate(nav_labels):
        x = 30 + i * 100
        d.ellipse([x, 725, x + 30, 755], fill=(70, 110, 230) if label == "마이" else (210, 210, 210))
        d.text((x - 10, 758), label, font=f_small, fill=(60, 60, 60))

    img.save(DATA_DIR / "figma_mock.png")


if __name__ == "__main__":
    make_product_image()
    make_receipt_image()
    make_contract_image()
    make_table_image()
    make_bar_chart()
    make_line_chart()
    make_dashboard()
    make_webpage_mock()
    make_figma_mock()
    print(f"샘플 이미지 9종 생성 완료 → {DATA_DIR}")

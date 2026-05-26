# -*- coding: utf-8 -*-
"""
타이타닉 생존률 분석 (titanic3 데이터셋, 승객 1,309명)
발표용 단계별 분석 스크립트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ──────────────────────────────────────────────────────────────
# 한글 폰트 설정  (seaborn 테마를 먼저 적용한 뒤 폰트를 지정해야
# 폰트 설정이 덮어써지지 않음)
# ──────────────────────────────────────────────────────────────
sns.set_style("whitegrid")

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT_PATH)
KR_FONT = fm.FontProperties(fname=FONT_PATH).get_name()  # 'Noto Sans CJK JP'
plt.rcParams["font.family"] = KR_FONT
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

# 색상 팔레트 (생존=초록, 사망=빨강 계열)
C_SURV, C_DIED = "#2a9d8f", "#e76f51"
C_RATE = "#264653"


def annotate_rate(ax, fmt="{:.1f}%", offset=1.5):
    """막대 위에 수치 라벨을 표시"""
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(fmt.format(h), (p.get_x() + p.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=9,
                        xytext=(0, offset), textcoords="offset points")


# ══════════════════════════════════════════════════════════════
# STEP 0. 데이터 불러오기 & 전처리
# ══════════════════════════════════════════════════════════════
df = pd.read_csv("/mnt/user-data/uploads/titanic.csv", index_col=0)

print("=" * 55)
print("STEP 0  데이터 개요")
print("=" * 55)
print(f"전체 승객 수 : {len(df):,}명")
print(f"전체 생존률  : {df['survived'].mean() * 100:.1f}%  "
      f"(생존 {df['survived'].sum()}명 / 사망 {(df['survived'] == 0).sum()}명)")
print(f"결측치 → 나이 {df['age'].isnull().sum()}건, "
      f"요금 {df['fare'].isnull().sum()}건, "
      f"탑승지 {df['embarked'].isnull().sum()}건")

# 결측치 처리: 나이는 등급+성별 중앙값, 요금은 등급 중앙값, 탑승지는 최빈값
df["age"] = df.groupby(["pclass", "sex"])["age"].transform(
    lambda s: s.fillna(s.median()))
df["fare"] = df.groupby("pclass")["fare"].transform(
    lambda s: s.fillna(s.median()))
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# 한글 라벨용 파생 변수
df["sex_kr"] = df["sex"].map({"male": "남성", "female": "여성"})
df["class_kr"] = df["pclass"].map({1: "1등석", 2: "2등석", 3: "3등석"})
df["embarked_kr"] = df["embarked"].map(
    {"S": "사우샘프턴(S)", "C": "셰르부르(C)", "Q": "퀸스타운(Q)"})


# ══════════════════════════════════════════════════════════════
# STEP 1. 성별 생존률
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("STEP 1  성별 생존률")
print("=" * 55)
sex_rate = df.groupby("sex_kr")["survived"].agg(["mean", "count"])
sex_rate["mean"] *= 100
print(sex_rate.round(1))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
fig.suptitle("STEP 1 · 성별 생존률", fontsize=15, fontweight="bold")

# (좌) 성별 생존/사망 인원
ct = pd.crosstab(df["sex_kr"], df["survived"])
ct.columns = ["사망", "생존"]
ct[["생존", "사망"]].plot(kind="bar", stacked=True, ax=ax[0],
                          color=[C_SURV, C_DIED], width=0.55)
ax[0].set_title("성별 생존·사망 인원")
ax[0].set_xlabel(""); ax[0].set_ylabel("인원 수")
ax[0].tick_params(axis="x", rotation=0)

# (우) 성별 생존률
bars = sns.barplot(x=sex_rate.index, y=sex_rate["mean"], ax=ax[1],
                   palette=[C_DIED, C_SURV], width=0.55)
ax[1].set_title("성별 생존률 (%)")
ax[1].set_xlabel(""); ax[1].set_ylabel("생존률 (%)")
ax[1].set_ylim(0, 100)
ax[1].axhline(df["survived"].mean() * 100, ls="--", color="gray", lw=1)
annotate_rate(ax[1])
plt.tight_layout()
plt.savefig("/home/claude/01_sex.png", bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════
# STEP 2. 가족 구성원별 생존률
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("STEP 2  가족 구성원별 생존률")
print("=" * 55)
# sibsp(형제·배우자) + parch(부모·자녀) + 본인 = 가족 규모
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["alone"] = np.where(df["family_size"] == 1, "혼자 탑승", "가족 동반")

fam = df.groupby("family_size")["survived"].agg(["mean", "count"])
fam["mean"] *= 100
print(fam.round(1))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
fig.suptitle("STEP 2 · 가족 구성원별 생존률", fontsize=15, fontweight="bold")

# (1) 가족 규모별 생존률
sns.barplot(x=fam.index, y=fam["mean"], ax=ax[0], color=C_RATE)
ax[0].set_title("가족 규모별 생존률")
ax[0].set_xlabel("가족 규모 (본인 포함, 명)"); ax[0].set_ylabel("생존률 (%)")
ax[0].set_ylim(0, 100)
annotate_rate(ax[0])

# (2) 형제·배우자(sibsp), 부모·자녀(parch) 수별 생존률
sib = df.groupby("sibsp")["survived"].mean() * 100
par = df.groupby("parch")["survived"].mean() * 100
ax[1].plot(sib.index, sib.values, "o-", color=C_SURV, label="형제·배우자(SibSp)")
ax[1].plot(par.index, par.values, "s-", color=C_DIED, label="부모·자녀(Parch)")
ax[1].set_title("동반 가족 유형별 생존률")
ax[1].set_xlabel("동반 인원 수"); ax[1].set_ylabel("생존률 (%)")
ax[1].set_ylim(0, 100); ax[1].legend()

# (3) 혼자 vs 가족 동반
al = df.groupby("alone")["survived"].mean() * 100
sns.barplot(x=al.index, y=al.values, ax=ax[2],
            palette=[C_SURV, C_DIED], width=0.5)
ax[2].set_title("혼자 vs 가족 동반")
ax[2].set_xlabel(""); ax[2].set_ylabel("생존률 (%)")
ax[2].set_ylim(0, 100)
annotate_rate(ax[2])
plt.tight_layout()
plt.savefig("/home/claude/02_family.png", bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════
# STEP 3. 탑승지 · 요금 · 좌석 등급별 생존률
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("STEP 3  탑승지 · 요금 · 좌석 등급별 생존률")
print("=" * 55)

# 요금을 4분위 구간으로 나눔
df["fare_band"] = pd.qcut(df["fare"], 4,
                          labels=["저가\n(하위25%)", "중저가", "중고가", "고가\n(상위25%)"])

print("[탑승지별]")
print((df.groupby("embarked_kr")["survived"].mean() * 100).round(1))
print("[등급별]")
print((df.groupby("class_kr")["survived"].mean() * 100).round(1))
print("[요금 구간별]")
print((df.groupby("fare_band", observed=True)["survived"].mean() * 100).round(1))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
fig.suptitle("STEP 3 · 탑승지 · 요금 · 좌석 등급별 생존률",
             fontsize=15, fontweight="bold")

# (1) 좌석 등급
cls = df.groupby("class_kr")["survived"].mean() * 100
sns.barplot(x=cls.index, y=cls.values, ax=ax[0],
            palette=["#2a9d8f", "#e9c46a", "#e76f51"], width=0.55)
ax[0].set_title("좌석 등급별 생존률")
ax[0].set_xlabel(""); ax[0].set_ylabel("생존률 (%)"); ax[0].set_ylim(0, 100)
annotate_rate(ax[0])

# (2) 탑승지
emb = df.groupby("embarked_kr")["survived"].mean() * 100
sns.barplot(x=emb.index, y=emb.values, ax=ax[1], color=C_RATE, width=0.55)
ax[1].set_title("탑승 항구별 생존률")
ax[1].set_xlabel(""); ax[1].set_ylabel("생존률 (%)"); ax[1].set_ylim(0, 100)
ax[1].tick_params(axis="x", rotation=12)
annotate_rate(ax[1])

# (3) 요금 구간
far = df.groupby("fare_band", observed=True)["survived"].mean() * 100
sns.barplot(x=far.index, y=far.values, ax=ax[2],
            palette="YlGnBu", width=0.6)
ax[2].set_title("요금 구간별 생존률")
ax[2].set_xlabel("표 가격 구간"); ax[2].set_ylabel("생존률 (%)")
ax[2].set_ylim(0, 100)
annotate_rate(ax[2])
plt.tight_layout()
plt.savefig("/home/claude/03_embarked_fare_class.png", bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════
# STEP 4. 나이별 생존률
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("STEP 4  나이별 생존률")
print("=" * 55)
bins = [0, 12, 18, 30, 45, 60, 100]
labels = ["어린이\n(0-12)", "청소년\n(13-18)", "청년\n(19-30)",
          "중년\n(31-45)", "장년\n(46-60)", "노년\n(60+)"]
df["age_band"] = pd.cut(df["age"], bins=bins, labels=labels)
age_rate = df.groupby("age_band", observed=True)["survived"].agg(["mean", "count"])
age_rate["mean"] *= 100
print(age_rate.round(1))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
fig.suptitle("STEP 4 · 나이별 생존률", fontsize=15, fontweight="bold")

# (좌) 연령대별 생존률
sns.barplot(x=age_rate.index, y=age_rate["mean"], ax=ax[0],
            palette="viridis", width=0.7)
ax[0].set_title("연령대별 생존률")
ax[0].set_xlabel(""); ax[0].set_ylabel("생존률 (%)"); ax[0].set_ylim(0, 100)
ax[0].axhline(df["survived"].mean() * 100, ls="--", color="gray", lw=1)
annotate_rate(ax[0])

# (우) 나이 분포 (생존 vs 사망)
for v, c, lab in [(1, C_SURV, "생존"), (0, C_DIED, "사망")]:
    sns.kdeplot(df.loc[df["survived"] == v, "age"], ax=ax[1],
                fill=True, alpha=0.4, color=c, label=lab)
ax[1].set_title("나이 분포: 생존 vs 사망")
ax[1].set_xlabel("나이"); ax[1].set_ylabel("밀도"); ax[1].legend()
plt.tight_layout()
plt.savefig("/home/claude/04_age.png", bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════
# STEP 5. 추가 분석 — 놓치기 쉬운 상관관계
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("STEP 5  추가 상관관계 분석")
print("=" * 55)

# 5-1. 이름에서 호칭(Title) 추출 — 사회적 지위/성별/연령의 압축 정보
df["title"] = df["name"].str.extract(r",\s*([^.]+)\.")
title_map = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Ms": "Miss", "Mlle": "Miss", "Mme": "Mrs",
    "Dr": "지위/직함", "Rev": "지위/직함", "Col": "지위/직함",
    "Major": "지위/직함", "Capt": "지위/직함", "Sir": "지위/직함",
    "Lady": "지위/직함", "the Countess": "지위/직함", "Don": "지위/직함",
    "Dona": "지위/직함", "Jonkheer": "지위/직함",
}
df["title_grp"] = df["title"].map(title_map).fillna("지위/직함")
ttl = df.groupby("title_grp")["survived"].mean() * 100
ttl = ttl.sort_values()
print("[호칭별 생존률]")
print(ttl.round(1))

# 5-2. 객실(Cabin) 정보 기록 여부 — 정보가 기록된 승객일수록 상위 등급
df["cabin_known"] = np.where(df["cabin"].notnull(), "객실정보 있음", "객실정보 없음")
print("[객실정보 기록 여부별 생존률]")
print((df.groupby("cabin_known")["survived"].mean() * 100).round(1))

fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
fig.suptitle("STEP 5 · 추가로 발견한 상관관계", fontsize=15, fontweight="bold")

# (1) 호칭별 생존률
sns.barplot(x=ttl.values, y=ttl.index, ax=ax[0, 0], palette="rocket", orient="h")
ax[0, 0].set_title("① 호칭(Title)별 생존률 — 이름에 숨은 정보")
ax[0, 0].set_xlabel("생존률 (%)"); ax[0, 0].set_ylabel(""); ax[0, 0].set_xlim(0, 100)
for p in ax[0, 0].patches:
    ax[0, 0].annotate(f"{p.get_width():.1f}%",
                      (p.get_width() + 1, p.get_y() + p.get_height() / 2),
                      va="center", fontsize=9)

# (2) 성별 × 좌석 등급 교차 생존률 (히트맵)
pivot = df.pivot_table("survived", "sex_kr", "class_kr", aggfunc="mean") * 100
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax[0, 1],
            vmin=0, vmax=100, cbar_kws={"label": "생존률 (%)"},
            linewidths=1, linecolor="white")
ax[0, 1].set_title("② 성별 × 좌석 등급 교차 생존률")
ax[0, 1].set_xlabel(""); ax[0, 1].set_ylabel("")
ax[0, 1].tick_params(axis="y", rotation=0)

# (3) 객실정보 기록 여부
cab = df.groupby("cabin_known")["survived"].mean() * 100
sns.barplot(x=cab.index, y=cab.values, ax=ax[1, 0],
            palette=[C_SURV, C_DIED], width=0.5)
ax[1, 0].set_title("③ 객실정보 기록 여부별 생존률")
ax[1, 0].set_xlabel(""); ax[1, 0].set_ylabel("생존률 (%)"); ax[1, 0].set_ylim(0, 100)
annotate_rate(ax[1, 0])

# (4) 주요 변수 상관관계 히트맵
df["is_female"] = (df["sex"] == "female").astype(int)
corr_cols = {
    "survived": "생존", "is_female": "여성", "pclass": "좌석등급",
    "age": "나이", "fare": "요금", "family_size": "가족규모",
    "sibsp": "형제·배우자", "parch": "부모·자녀",
}
corr = df[list(corr_cols)].corr().rename(index=corr_cols, columns=corr_cols)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            ax=ax[1, 1], vmin=-1, vmax=1, linewidths=0.5, linecolor="white")
ax[1, 1].set_title("④ 주요 변수 간 상관계수")
ax[1, 1].tick_params(axis="y", rotation=0)
ax[1, 1].tick_params(axis="x", rotation=35)
for lbl in ax[1, 1].get_xticklabels():
    lbl.set_ha("right")
plt.tight_layout()
plt.savefig("/home/claude/05_extra.png", bbox_inches="tight")
plt.close()

print("\n생존과의 상관계수 (절댓값 큰 순):")
print(corr["생존"].drop("생존").abs().sort_values(ascending=False).round(3))
print("\n✓ 모든 그래프 저장 완료")

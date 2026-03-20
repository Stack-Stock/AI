# 📊 STACK&STOCK PIPELINE

> 뉴스와 주가 데이터를 기반으로
> **투자 판단을 연습하는 게임용 데이터 생성 시스템**

---

## 📌 Overview

이 프로젝트는 실제 뉴스 기사와 주가 데이터를 활용하여
**투자 이벤트 → 원인 기사 → 게임 시나리오**로 변환하는
엔드투엔드 데이터 파이프라인이다.

단순 데이터 수집이 아니라,
**“이 가격 변동의 원인이 되는 뉴스는 무엇인가?”**를 추론하는 데 초점을 둔다.

---

## 🏗️ Pipeline

```
STEP1  뉴스 정제 및 구조화
STEP2  주가 이벤트 생성
STEP3  기사–이벤트 매칭 (임베딩 + 재랭킹)
STEP4  게임 케이스 선정
STEP5  시나리오 및 시장 데이터 생성
```

---

## ⚙️ Key Features

### 1. 📄 News Processing

* 기사 정제 및 구조화
* 산업군 분류 + 기업 매칭
* 기사 품질 스코어링

### 2. 📈 Price Event Detection

* 급등/급락/거래량 이벤트 탐지
* 이벤트 강도 계산
* 중복 이벤트 클러스터링

### 3. 🤖 Hybrid Matching

* Voyage 임베딩 기반 유사도 계산
* 규칙 기반 재랭킹 결합
* 이벤트–기사 정합성 강화

### 4. 🎮 Game Data Selection

* 상승/하락 비율 조정 (60:40)
* 종목 편중 방지
* 고품질 케이스 필터링

### 5. 🧠 Scenario Generation

* Claude 기반 시나리오 생성
* 매체별 정보 밀도 차등 (phone / tv / newspaper)
* 시장 데이터 시뮬레이션

---

## 🧠 Core Techniques

* Embedding: `Voyage AI`
* LLM: `Claude Sonnet 4.5`
* Matching: `Cosine Similarity + Rule-based Reranking`
* Simulation:

  * Gaussian Noise
  * Sector Spillover
  * Tanh Scaling

---

## 🚨 Challenges & Solutions

| 문제            | 해결                    |
| ------------- | --------------------- |
| 기사–이벤트 연결 부정확 | 임베딩 + 규칙 기반 하이브리드 매칭  |
| 이벤트 원인 해석 불가  | 가격/원인 분리 구조           |
| LLM 환각        | 생성 제한 + 검증 + fallback |
| 게임 밸런스 문제     | 분포 제어 + 수익률 보정        |

---

## ✅ Result

* 뉴스 기반 **투자 이벤트 데이터셋 구축**
* **의미 기반 기사–이벤트 매칭 시스템 설계**
* 게임용 **시나리오 + 시장 데이터 자동 생성**

---

## 💡 Summary

> 뉴스 → 시장 → 게임을 연결하는
> **투자 학습용 데이터 파이프라인** 🚀

---


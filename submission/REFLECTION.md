# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Vu Hai Dang
**Khoá:** K4
**Tier đã chạy:** T4
**Ngày:** 2026-01-XX

> ⚠️ **Lưu ý:** GPU Colab hết phiên ở bước NB4 (so sánh SFT vs SFT+DPO). Các số liệu §1-§3 từ `dpo_metrics.json` và ảnh từ NB1-NB3. Phần NB4 và §4 không có dữ liệu do mất phiên.

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | **70.8%** (từ ảnh 02b-pref-length.png) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | Chưa chạy được (GPU hết phiên ở NB4) |
| Chi phí | 0 đồng (Colab miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---:|---:|
| Thời gian huấn luyện NB3 | _<từ notebook>_ phút |
| VRAM cao nhất | _<từ notebook>_ MB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0.0907 |
| Độ chính xác reward trên held-out | 0.68 (68%) |
| Margin trên held-out | 0.0874 |
| Chẩn đoán tự động (`diagnosis`) | **INTENDED** |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | Chưa có (GPU hết phiên) |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

**Quan sát từ ảnh `03-dpo-reward-curves.png`:**

Trên **tập huấn luyện (train)**:
- `rewards/chosen` tăng: từ ~0.1 lên ~0.4
- `rewards/rejected` giảm: từ ~0.1 xuống ~0.3
- Margin (hiệu số) tăng rõ ràng theo thời gian huấn luyện

Trên **tập held-out (eval)**:
- `rewards/chosen` tăng: từ ~0.1 lên ~0.4
- `rewards/rejected` giảm: từ ~0.1 xuống ~0.3
- **Đi cùng hướng với tập huấn luyện** → không có dấu hiệu overfit nghiêm trọng

**Phân tích:**
- Margin tăng là do **chosen tăng** và **rejected giảm** đồng thời → đây là **INTENDED**, đúng kỳ vọng lý thuyết DPO
- Không có hiện tượng **likelihood displacement** (chosen giảm trong khi margin tăng)
- Đường eval đi cùng hướng với train → mô hình không chỉ học thuộc tập huấn luyện
- Accuracy 68% trên held-out (từ `dpo_metrics.json`) cho thấy mô hình phân biệt chosen/rejected tốt hơn ngẫu nhiên (50%)

Chẩn đoán **INTENDED** khớp hoàn toàn với những gì tôi thấy trên biểu đồ.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

⚠️ **Không có dữ liệu** — GPU Colab hết phiên ở NB4.

_Không có dữ liệu từ NB4 do GPU hết phiên. Win rate, khoảng tin cậy 95%, sanity accuracy và các ví dụ cụ thể không có sẵn._

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | 0.087 | 0.68 | INTENDED | Baseline |
| 0.5 | | | | |

_Nếu không chạy: viết giả thuyết 3 câu về điều bạn dự đoán sẽ thấy._

Giả thuyết:
- **β càng nhỏ (0.05):** Penalty mạnh hơn → margin tăng nhanh hơn, nhưng có thể gây mất ổn định hoặc collapse
- **β càng lớn (0.5):** Conservative hơn, gần với reference model, thay đổi ít hơn
- Với β=0.1 đã đạt INTENDED và 68% accuracy → baseline hợp lý

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

**Quyết định của tôi: Sử dụng β = 0.1 (mặc định)**

**Phương án thay thế:**
- β = 0.05 (nhỏ hơn → penalty mạnh hơn, mô hình thay đổi nhanh hơn)
- β = 0.5 (lớn hơn → conservative, gần với reference model hơn)

**Vì sao chọn β = 0.1:**
Tôi giữ nguyên giá trị mặc định vì:
1. Đây là baseline được khuyến nghị trong nhiều paper về DPO (như paper gốc của DPO từ Stanford)
2. An toàn cho lần chạy đầu tiên — tránh hiện tượng collapse khi penalty quá mạnh
3. Kết quả cho thấy INTENDED với accuracy 68% trên held-out → phù hợp để bắt đầu

**Kết quả:**
Với β = 0.1, chẩn đoán INTENDED cho thấy mô hình học đúng hướng. Accuracy 68% trên held-out cao hơn ngẫu nhiên (50%) nhưng chưa cao, cho thấy có thể cần huấn luyện thêm epochs hoặc thử β khác.

**Nếu làm lại:**
Tôi sẽ chạy β-sweep (0.05, 0.1, 0.5) để so sánh systematic. Đặc biệt, β = 0.05 có thể cho margin cao hơn nhưng cần theo dõi sự ổn định trên held-out. Ngoài ra, tôi sẽ chia thời gian tải kết quả sau mỗi bước thay vì chờ hết tất cả.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

_Không chạy được (phần bonus)._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

_Không chạy được (phần bonus)._

---

## 9. GRPO (bonus NB7)

_Không chạy được (phần bonus)._

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

GPU Colab hết phiên ở NB4 — bước quan trọng nhất để đánh giá DPO có thật sự tốt hơn SFT hay không. Bài học: luôn tải kết quả về ngay sau mỗi bước, không chờ chạy hết tất cả rồi mới tải.

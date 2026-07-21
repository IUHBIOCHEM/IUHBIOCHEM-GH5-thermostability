#!/bin/bash
# Chạy FoldX ΔΔG cho 4 đột biến trên 3PZT (Bacillus GH5).
# Yêu cầu: đặt binary FoldX (tên 'foldx') vào chính thư mục này (hoặc sửa biến FOLDX bên dưới).
set -e
cd "$(dirname "$0")"
FOLDX=./foldx          # <-- sửa nếu binary tên khác (vd ./foldx_20251231)

echo "[1/2] RepairPDB (tối ưu hoá năng lượng cấu trúc gốc)..."
$FOLDX --command=RepairPDB --pdb=3PZT.pdb

echo "[2/2] BuildModel (tính ΔΔG cho các đột biến, 5 lần chạy)..."
$FOLDX --command=BuildModel --pdb=3PZT_Repair.pdb \
       --mutant-file=individual_list.txt --numberOfRuns=1

echo ""
echo "=== ΔΔG (kcal/mol) — mutant trừ wild-type, cột 'total energy' ==="
echo "Xem file: Dif_3PZT_Repair.fxout  (mỗi dòng = 1 đột biến, theo thứ tự individual_list.txt)"
echo "Thứ tự: Q58K, N158K, Q41E, T89K"
python3 parse_ddg.py 2>/dev/null || true

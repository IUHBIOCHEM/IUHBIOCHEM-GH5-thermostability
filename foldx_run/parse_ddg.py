# Đọc Dif_3PZT_Repair.fxout -> ΔΔG mỗi đột biến
import glob, csv
muts=[l.strip().rstrip(';') for l in open('individual_list.txt') if l.strip()]
f=glob.glob('Dif_*_Repair.fxout')
if not f:
    print('Chưa có file Dif_*.fxout — chạy BuildModel trước.'); raise SystemExit
lines=[l for l in open(f[0]) if l.strip() and not l.startswith(('*','FoldX','Pdb'))]
# tìm dòng dữ liệu (bắt đầu bằng tên pdb)
rows=[l.split('\t') for l in open(f[0]) if l.startswith('3PZT')]
print(f'{"Mutation":<10}{"ddG (kcal/mol)":>16}  interpretation')
for m,r in zip(muts,rows):
    ddg=float(r[1])
    tag='STABILIZING' if ddg<-0.5 else ('destabilizing' if ddg>0.5 else 'neutral')
    print(f'{m:<10}{ddg:>16.2f}  {tag}')
print('\n(ΔΔG < 0 = ổn định hơn / tăng bền nhiệt; > 0 = kém ổn định. |ΔΔG|<0.5 ~ nhiễu)')

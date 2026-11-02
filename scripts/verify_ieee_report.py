import os
import sys
import docx
import re

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


docx_path = r'c:\MINE\PROJECTS\IVP\reports\SentinelPlate_IEEE_Paper_Clean.docx'
html_path = r'c:\MINE\PROJECTS\IVP\reports\SentinelPlate_IEEE_TwoColumn_Print.html'
fig_dir = r'c:\MINE\PROJECTS\IVP\reports\figures'

print('=== 1. CHECK FIGURES ===')
for f in ['fig1_enhancement.png', 'fig2_edge_iou.png', 'fig3_motion_estimation.png', 'fig4_pipeline_stages.png']:
    fp = os.path.join(fig_dir, f)
    print(f'  {f}: exists={os.path.exists(fp)}, size={os.path.getsize(fp):,} bytes')

print('\n=== 2. CHECK OUTPUT A (.DOCX) ===')
doc = docx.Document(docx_path)
print(f'  Paragraphs: {len(doc.paragraphs)}')
print(f'  Tables: {len(doc.tables)}')

span_p = sum(len(re.findall(r'\[span_\d+\]', p.text)) for p in doc.paragraphs)
span_t = sum(sum(len(re.findall(r'\[span_\d+\]', c.text)) for c in row.cells) for t in doc.tables for row in t.rows)
print(f'  Span tags in paragraphs: {span_p}, in tables: {span_t}')

works_cited = any('works cited' in p.text.lower() for p in doc.paragraphs)
print(f'  Trailing Works cited found: {works_cited}')

refs = [p.text for p in doc.paragraphs if p.text.strip().startswith('[') and p.text.strip()[1:3].strip(']').isdigit()]
print(f'  IEEE References count: {len(refs)} (first: {refs[0][:30]}..., last: {refs[-1][:30]}...)')

for idx, tbl in enumerate(doc.tables):
    first_row = [c.text.strip() for c in tbl.rows[0].cells]
    print(f'  Table {idx}: {len(tbl.rows)} rows x {len(tbl.columns)} cols -> {first_row[:2]}')

doc_images = 0
for rel in doc.part.rels.values():
    if 'image' in rel.target_ref:
        doc_images += 1
print(f'  Embedded images in docx: {doc_images}')

print('\n=== 3. CHECK OUTPUT B (.HTML) ===')
with open(html_path, 'r', encoding='utf-8') as f:
    html_content = f.read()
print(f'  HTML size: {len(html_content):,} chars')
span_in_html = len(re.findall(r'\[span_\d+\]', html_content))
print(f'  Span tags in HTML: {span_in_html}')
print(f'  Works cited in HTML: {"works cited" in html_content.lower()}')
print(f'  MathJax 3 included: {"mathjax" in html_content.lower()}')
print(f'  Figures embedded in HTML: {html_content.count("<img ")}')
print(f'  Tables embedded in HTML: {html_content.count("<table ")}')

import docx
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

doc = docx.Document()
tbl = doc.add_table(rows=2, cols=2)
hdr_cells = tbl.rows[0].cells
hdr_cells[0].text = 'Header 1'
hdr_cells[1].text = 'Header 2'

shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="1F4E79"/>')
hdr_cells[0]._tc.get_or_add_tcPr().append(shd)

doc.save(r'c:\MINE\PROJECTS\IVP\scratch\test_tbl.docx')
print('Table created successfully!')

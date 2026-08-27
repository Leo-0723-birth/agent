import zipfile, sys
from xml.etree import ElementTree as ET

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
path = r"C:/Users/86130/Desktop/06-智能风控与量化建模赛道-东吴证券-基于 Agentic AI 的上市公司监管问询概率预测与扫雷预警算法探索.docx"

with zipfile.ZipFile(path) as z:
    xml = z.read('word/document.xml')
root = ET.fromstring(xml)
body = root.find(f'{{{W}}}body')

def para_text(p):
    return ''.join(t.text or '' for t in p.iter(f'{{{W}}}t'))

out = []
for elem in body:
    tag = elem.tag.split('}')[-1]
    if tag == 'p':
        txt = para_text(elem)
        style = ''
        pPr = elem.find(f'{{{W}}}pPr')
        if pPr is not None:
            ps = pPr.find(f'{{{W}}}pStyle')
            if ps is not None:
                style = ps.get(f'{{{W}}}val', '')
        prefix = f'[{style}] ' if (style.startswith('Heading') or style.startswith('Title')) else ''
        if txt.strip():
            out.append(prefix + txt)
    elif tag == 'tbl':
        out.append('--- TABLE ---')
        for row in elem.findall(f'{{{W}}}tr'):
            cells = []
            for cell in row.findall(f'{{{W}}}tc'):
                ctext = ' '.join(para_text(p) for p in cell.findall(f'{{{W}}}p'))
                cells.append(ctext)
            out.append(' | '.join(cells))
        out.append('--- END TABLE ---')

full = '\n'.join(out)
with open(r'D:\competition_agent\scripts\docx_content.txt', 'w', encoding='utf-8') as f:
    f.write(full)
print(f"Extracted {len(full)} chars, {len(out)} lines")

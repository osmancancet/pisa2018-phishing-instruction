"""Build manuscript/main.docx from manuscript/main.md with pandoc + APA 7 CSL, then tidy it."""
import subprocess
import copy
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

M = 'manuscript'
ref = f'{M}/reference.docx'
# embed figures without ancillary metadata chunks (keep only image data)
import tempfile, struct, glob as _glob, os as _os
_figdir = tempfile.mkdtemp()
for _f in _glob.glob(f'{M}/*.png'):
    _d = open(_f, 'rb').read(); _out = [_d[:8]]; _i = 8
    while _i < len(_d):
        _n = struct.unpack('>I', _d[_i:_i+4])[0]; _t = _d[_i+4:_i+8]
        if _t in (b'IHDR', b'PLTE', b'IDAT', b'IEND', b'pHYs', b'tRNS', b'gAMA', b'sRGB'):
            _out.append(_d[_i:_i+12+_n])
        _i += 12 + _n
    open(_os.path.join(_figdir, _os.path.basename(_f)), 'wb').write(b''.join(_out))
subprocess.check_call(['pandoc', '-o', ref, '--print-default-data-file', 'reference.docx'])
d = Document(ref)
for st in d.styles:
    try:
        f = st.font
    except AttributeError:
        continue
    if f is None:
        continue
    f.name = 'Times New Roman'
    f.color.rgb = RGBColor(0, 0, 0)
    rpr = st.element.get_or_add_rPr()
    fonts = rpr.find(qn('w:rFonts'))
    if fonts is not None:
        for a in ('w:asciiTheme', 'w:hAnsiTheme', 'w:eastAsiaTheme', 'w:cstheme'):
            fonts.attrib.pop(qn(a), None)
    col = rpr.find(qn('w:color'))
    if col is not None:
        col.attrib.pop(qn('w:themeColor'), None)
    if st.name in ('Hyperlink',):
        f.underline = False
SBYN = {s.name: s for s in d.styles}
for name in ('Normal', 'Body Text', 'First Paragraph', 'Compact', 'Abstract', 'Bibliography'):
    if name in SBYN:
        st = SBYN[name]
        st.font.size = Pt(12)
        pf = st.paragraph_format
        pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
        pf.space_before = Pt(0); pf.space_after = Pt(6)
for name, size in (('Heading 1', 14), ('Heading 2', 12), ('Heading 3', 12), ('Title', 16)):
    if name not in SBYN:
        continue
    st = SBYN[name]; st.font.size = Pt(size); st.font.bold = True; st.font.italic = False
sec = d.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
for m in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'):
    setattr(sec, m, Cm(2.5))
ln = OxmlElement('w:lnNumType'); ln.set(qn('w:countBy'), '1'); ln.set(qn('w:restart'), 'continuous')
sec._sectPr.append(ln)
# footer page number
p = sec.footer.paragraphs[0] if sec.footer.paragraphs else sec.footer.add_paragraph()
pPr = p._p.get_or_add_pPr(); sl = OxmlElement('w:suppressLineNumbers'); pPr.append(sl)
p.alignment = 2
r = p.add_run()
for tag, txt in (('begin', None), (None, 'PAGE'), ('end', None)):
    if tag:
        e = OxmlElement('w:fldChar'); e.set(qn('w:fldCharType'), tag); r._r.append(e)
    else:
        e = OxmlElement('w:instrText'); e.set(qn('xml:space'), 'preserve'); e.text = txt; r._r.append(e)
d.save(ref)

subprocess.check_call(['pandoc', f'{M}/main.md', '-o', f'{M}/main.docx', '--citeproc',
                       f'--bibliography={M}/refs.bib', f'--csl={M}/apa.csl', f'--reference-doc={ref}',
                       f'--resource-path={_figdir}:{M}', '-M', 'link-citations=false'])
doc = Document(f'{M}/main.docx')
# tables: 9 pt single spaced, full width
for t in doc.tables:
    tblPr = t._tbl.tblPr
    w = tblPr.find(qn('w:tblW'))
    if w is None:
        w = OxmlElement('w:tblW'); tblPr.append(w)
    w.set(qn('w:w'), '5000'); w.set(qn('w:type'), 'pct')
    import re as _re
    for ri, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement('w:cantSplit'); trPr.append(cs)
        if ri == 0:
            th = OxmlElement('w:tblHeader'); trPr.append(th)
        panel = any(_re.match(r'^[A-C]\. ', c.text.strip()) for c in row.cells)
        for cell in row.cells:
            if panel or ri == 0:
                for par in cell.paragraphs:
                    par.paragraph_format.keep_with_next = True
            for par in cell.paragraphs:
                par.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                par.paragraph_format.space_after = Pt(0)
                for run in par.runs:
                    run.font.size = Pt(9)
for par in doc.paragraphs:
    for run in par.runs:
        run.font.color.rgb = RGBColor(0, 0, 0)
doc.core_properties.author = 'Osman Can Çetlenbik'
doc.core_properties.last_modified_by = 'Osman Can Çetlenbik'
doc.save(f'{M}/main.docx')
print('docx built')

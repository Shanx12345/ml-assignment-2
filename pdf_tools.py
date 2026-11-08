"""Small Markdown-to-PDF renderer used by the learning guide and submission utility."""
import re
import textwrap
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (Image, PageBreak, Paragraph, Preformatted,
                               SimpleDocTemplate, Spacer, Table, TableStyle)

WIDTH = A4[0] - 100
STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle(name='BodyRef', fontName='Helvetica', fontSize=10,
                         leading=13.5, spaceAfter=6.5, splitLongWords=True))
STYLES.add(ParagraphStyle(name='CellRef', fontName='Helvetica', fontSize=8,
                         leading=11, spaceAfter=0, splitLongWords=True))
STYLES.add(ParagraphStyle(name='CodeRef', fontName='Courier', fontSize=8,
                         leading=10, leftIndent=8, rightIndent=8, spaceAfter=9))
for name in ['Title', 'Heading1', 'Heading2', 'Heading3']:
    STYLES[name].textColor = colors.black
STYLES['Title'].fontSize, STYLES['Title'].leading = 24, 29
STYLES['Heading1'].fontSize, STYLES['Heading1'].leading = 17, 21
STYLES['Heading2'].fontSize, STYLES['Heading2'].leading = 13, 17
STYLES['Heading3'].fontSize, STYLES['Heading3'].leading = 11, 14


def inline(text):
    value = escape(text)
    value = re.sub(r'\[([^\]]+)\]\((https?://[^\s)]+)\)',
                   r'<link href="\2" color="#006B5E">\1</link>', value)
    value = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', value)
    value = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', value)
    return value


def paragraph(text, style='BodyRef'):
    return Paragraph(inline(text), STYLES[style])


def markdown_story(markdown):
    story, lines, i = [], markdown.splitlines(), 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == '<PAGEBREAK>':
            story.append(PageBreak())
            i += 1
            continue
        if line.startswith('```'):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code += textwrap.wrap(lines[i].expandtabs(4), width=94,
                                      subsequent_indent='    ', replace_whitespace=False,
                                      drop_whitespace=False) or ['']
                i += 1
            story.append(Preformatted('\n'.join(code), STYLES['CodeRef']))
            i += 1
            continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\-\s]+', c) for c in cells):
                    rows.append(cells)
                i += 1
            ncols = len(rows[0])
            widths = ([118] + [(WIDTH-118)/(ncols-1)]*(ncols-1) if ncols >= 6
                      else [116, WIDTH-116] if ncols == 2
                      else [WIDTH/ncols]*ncols)
            data = [[paragraph(c, 'CellRef') for c in row] for row in rows]
            table = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E7EDF0')),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LINEBELOW', (0,0), (-1,0), .6, colors.HexColor('#78909C')),
                ('LINEBELOW', (0,1), (-1,-1), .25, colors.HexColor('#D4DCE1')),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ]))
            story.extend([table, Spacer(1,10)])
            continue
        if line.startswith('#'):
            count = len(line) - len(line.lstrip('#'))
            style = 'Title' if count == 1 else 'Heading1' if count == 2 else 'Heading2' if count == 3 else 'Heading3'
            story.append(paragraph(line[count:].strip(), style))
            i += 1
            continue
        chunks = [line]
        i += 1
        if not re.match(r'^(- |\d+\. )', line):
            while i < len(lines) and lines[i].strip() and not re.match(r'^(#|\||```|<PAGEBREAK>|- |\d+\. )', lines[i].strip()):
                chunks.append(lines[i].strip())
                i += 1
        story.append(paragraph(' '.join(chunks)))
    return story


def make_pdf(path, story, title):
    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#526575'))
        canvas.drawString(50, 29, title)
        canvas.drawRightString(A4[0]-50, 29, str(doc.page))
        canvas.restoreState()
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=50, rightMargin=50,
                           topMargin=43, bottomMargin=48, title=title)
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)

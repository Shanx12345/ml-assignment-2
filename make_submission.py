"""Assemble the mandatory PDF after your real links and BITS screenshot exist."""
import argparse
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image as PillowImage
from reportlab.platypus import Image, PageBreak, Spacer

from pdf_tools import WIDTH, make_pdf, markdown_story, paragraph
from shopper_models import ROOT


def validate_url(value, kind):
    parts = urlsplit(value)
    if parts.scheme != 'https' or parts.username or parts.password:
        raise ValueError(f'{kind}: supply a real HTTPS URL.')
    if kind == 'GitHub':
        path = parts.path.strip('/').split('/')
        if parts.hostname != 'github.com' or len(path) != 2:
            raise ValueError('GitHub URL must identify one repository, not a profile or file.')
    elif not (parts.hostname or '').endswith('.streamlit.app'):
        raise ValueError('Supply the deployed Community Cloud URL ending in .streamlit.app.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-url', required=True)
    parser.add_argument('--app-url', required=True)
    parser.add_argument('--screenshot', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--student-id', required=True)
    parser.add_argument('--readme', type=Path, default=ROOT/'README.md')
    parser.add_argument('--output', type=Path, default=ROOT/'submission.pdf')
    args = parser.parse_args()
    validate_url(args.repo_url, 'GitHub')
    validate_url(args.app_url, 'Streamlit')
    if not args.screenshot.is_file():
        raise ValueError('A genuine BITS execution screenshot is required.')
    text = args.readme.read_text()
    for heading in ['a. Problem statement', 'b. Dataset description',
                    'c. Github Repository Link', 'd. Models used']:
        if heading not in text:
            raise ValueError('README is missing required section: ' + heading)
    if args.repo_url not in text or args.app_url not in text:
        raise ValueError('Update README with the exact supplied repository and app URLs first.')
    story = [paragraph('Machine Learning Assignment 2', 'Title'),
             paragraph(args.name + ' | ' + args.student_id),
             paragraph('1 GitHub Repository Link', 'Heading1'),
             paragraph(f'[{args.repo_url}]({args.repo_url})'),
             paragraph('2 Live Streamlit App Link', 'Heading1'),
             paragraph(f'[{args.app_url}]({args.app_url})'),
             PageBreak(), paragraph('3 BITS Virtual Lab execution screenshot', 'Heading1')]
    with PillowImage.open(args.screenshot) as image:
        w, h = image.size
    factor = min(WIDTH/w, 620/h)
    story += [Image(str(args.screenshot), width=w*factor, height=h*factor),
              Spacer(1,10), paragraph('Screenshot supplied from the actual assignment execution on BITS Virtual Lab.'),
              PageBreak(), paragraph('4 GitHub README content', 'Heading1')]
    story += markdown_story(text)
    make_pdf(args.output, story, 'Machine Learning Assignment 2')
    print(args.output.resolve())
    print('PDF assembled. Check both links in a signed-out browser and inspect every page before submitting.')


if __name__ == '__main__':
    main()

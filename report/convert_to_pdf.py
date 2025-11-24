# Convert Markdown Report to PDF
# Install required package first: pip install markdown-pdf

import markdown
from weasyprint import HTML
import os

def convert_md_to_pdf(md_file, pdf_file):
    """Convert markdown file to PDF"""
    
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Add CSS styling
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                page-break-before: always;
            }}
            h2 {{
                color: #34495e;
                border-bottom: 2px solid #95a5a6;
                padding-bottom: 5px;
                margin-top: 30px;
            }}
            h3 {{
                color: #7f8c8d;
                margin-top: 25px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                page-break-inside: avoid;
            }}
            table, th, td {{
                border: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
                padding: 12px;
                text-align: left;
            }}
            td {{
                padding: 10px;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            pre {{
                background-color: #2c3e50;
                color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                page-break-inside: avoid;
            }}
            pre code {{
                background-color: transparent;
                color: #ecf0f1;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                padding-left: 20px;
                margin-left: 0;
                color: #555;
                font-style: italic;
            }}
            .page-break {{
                page-break-after: always;
            }}
            @page {{
                size: A4;
                margin: 2cm;
            }}
        </style>
    </head>
    <body>
        {}
    </body>
    </html>
    """
    
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
    full_html = html_template.format(html_content)
    
    # Convert HTML to PDF
    HTML(string=full_html).write_pdf(pdf_file)
    print(f"✅ PDF created successfully: {pdf_file}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_file = os.path.join(script_dir, "report.md")
    pdf_file = os.path.join(script_dir, "report.pdf")
    
    print("📄 Converting Markdown to PDF...")
    print(f"   Input: {md_file}")
    print(f"   Output: {pdf_file}")
    
    try:
        convert_md_to_pdf(md_file, pdf_file)
        print("\n✅ Conversion complete!")
        print(f"\n📁 PDF location: {pdf_file}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nAlternative methods:")
        print("1. Use online converter: https://www.markdowntopdf.com/")
        print("2. Use VS Code extension: 'Markdown PDF'")
        print("3. Use Pandoc: pandoc report.md -o report.pdf")

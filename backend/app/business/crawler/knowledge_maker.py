import mimetypes
import re
import sqlite3
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

import fitz  # PyMuPDF のモジュール名
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

console = Console()


def get_domain(url):
    return urlparse(url).netloc


def get_file_extension(url, content_type):
    if content_type and content_type != "other":
        ext = mimetypes.guess_extension(content_type)
        if ext:
            return ext

    path = urlparse(url).path
    ext = Path(path).suffix

    return ext if ext else ".bin"


def sanitize_filename(name):
    # ファイルシステムで許可されない文字を置換
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def generate_unique_filename(url, content_type):
    parsed_url = urlparse(url)
    path = unquote(parsed_url.path)
    query = unquote(parsed_url.query)

    # パスとクエリからファイル名を生成
    filename = path.strip("/").replace("/", "_")
    if query:
        filename += "_" + query.replace("&", "_")

    # 文字数制限（例：255文字）を超える場合、UUID4を使用
    if len(filename) > 255:
        unique_id = str(uuid.uuid4())
        filename = f"{filename[:240]}_{unique_id}"

    # ファイル名をサニタイズ
    filename = sanitize_filename(filename)

    # 適切な拡張子を追加
    extension = get_file_extension(url, content_type)
    if not filename.endswith(extension):
        filename += extension

    return filename


def split_pdf_by_size(input_pdf_path, size_limit_mb):
    size_limit = size_limit_mb * 1024 * 1024  # サイズ制限をMBからバイトに変換

    doc = fitz.open(input_pdf_path)
    total_pages = doc.page_count
    print(f"入力PDFの総ページ数: {total_pages}")

    virtual_doc = fitz.open()
    part = 1
    page_infos = []

    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        virtual_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        page_infos.append((page_num, page))

        # 現在のPDFをメモリ上のバッファに書き込み、サイズを確認
        temp_buffer = virtual_doc.write()
        temp_size = len(temp_buffer)

        if temp_size <= size_limit:
            continue

        # サイズが制限を超えた場合、最後のページを削除して保存
        _ = page_infos.pop()

        ## 最後に追加したページを除いて保存
        try:
            n_pages = [n for n, _ in page_infos]
            page_s, page_e = n_pages[0], n_pages[-1]
            output_doc = fitz.open()
            output_doc.insert_pdf(doc, from_page=page_s, to_page=page_e)
        except Exception as e:
            print(f"最後のページを削除できませんでした: {e}")

        ## サイズを再計算
        temp_buffer = output_doc.write()
        temp_size = len(temp_buffer)

        ## 現在のPDFを保存
        ### 出力ファイルパス output_pdf_path を input_pdf_path の拡張子を除いた部分に変更
        output_pdf_path = input_pdf_path.with_suffix(f".part_{part:03d}.pdf")
        output_doc.save(output_pdf_path)
        print(f"{output_pdf_path} を作成しました。サイズ: {temp_size} バイト")

        part += 1
        ## 新しいドキュメントを開始し、削除したページから再開
        virtual_doc = fitz.open()
        virtual_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        page_infos = [(page_num, page)]

    # 残りのページを保存
    if virtual_doc.page_count > 0:
        # 出力ファイルパス output_pdf_path を input_pdf_path の拡張子を除いた部分に変更
        output_pdf_path = input_pdf_path.with_suffix(f".part_{part:03d}.pdf")
        virtual_doc.save(output_pdf_path)
        print(f"{output_pdf_path} を作成しました。サイズ: {len(virtual_doc.write())} バイト")


def save_content(folder: Path, filename: str, content: bytes):
    file_path = folder / filename
    file_path.write_bytes(content)

    # file_pathがPDFの場合、サイズが10MBを超える場合に分割
    if file_path.suffix == ".pdf":
        split_pdf_by_size(file_path, size_limit_mb=10)
        # 元のファイルを削除
        file_path.unlink()


def create_mapping_table(conn):
    cursor = conn.cursor()
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS url_file_mapping (
        url TEXT PRIMARY KEY,
        file_path TEXT NOT NULL
    )
    """
    )
    conn.commit()


def save_mapping(conn, url, file_path):
    cursor = conn.cursor()
    cursor.execute(
        """
    INSERT OR REPLACE INTO url_file_mapping (url, file_path) VALUES (?, ?)
    """,
        (url, file_path),
    )
    conn.commit()


def export_data(db_path: Path, output_dir: Path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    create_mapping_table(conn)

    cursor.execute("SELECT url, domain, content, content_type FROM pages")
    records = cursor.fetchall()

    output_dir.mkdir(exist_ok=True)

    with Progress() as progress:
        task = progress.add_task("[green]Exporting data...", total=len(records))
        for url, domain, content, content_type in records:
            domain_folder = output_dir / domain
            domain_folder.mkdir(parents=True, exist_ok=True)

            filename = generate_unique_filename(url, content_type)

            save_content(domain_folder, filename, content)

            file_path = str(domain_folder / filename)
            save_mapping(conn, url, file_path)

            progress.update(task, advance=1)

    cursor.execute("SELECT COUNT(*) FROM url_file_mapping")
    mapping_count = cursor.fetchone()[0]

    conn.close()

    console.print(Panel.fit("Export completed successfully!", title="Status", border_style="green"))
    console.print(f"Data saved in: [cyan]{output_dir}[/cyan]")
    console.print(f"Total URL to filename mappings saved: [cyan]{mapping_count}[/cyan]")

    # Display a summary table
    table = Table(title="Export Summary")
    table.add_column("Description", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Output Directory", str(output_dir))
    table.add_row("Total Mappings", str(mapping_count))
    table.add_row("Database File", str(db_path))

    console.print(table)


def main(
    db_path: Path = typer.Argument(..., help="Path to the SQLite database file"),
    output_dir: Path = typer.Option(
        Path("knowledge"), "--output", "-o", help="Directory to save exported files (default: ./knowledge)"
    ),
):
    """
    Export data from a SQLite database to files organized by domain.
    """
    if not db_path.exists():
        console.print(f"[bold red]Error:[/bold red] Database file {db_path} does not exist.", err=True)
        raise typer.Exit(code=1)

    export_data(db_path, output_dir)


if __name__ == "__main__":
    typer.run(main)

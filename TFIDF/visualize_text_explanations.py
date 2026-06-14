import argparse
import csv
import html
import json
import re
import sys
from pathlib import Path


TFIDF_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = TFIDF_ROOT.parent
DEFAULT_VIS_DIR = TFIDF_ROOT / 'vis'
DEFAULT_OUTPUT_DIR = DEFAULT_VIS_DIR / 'text_highlights'

METHOD_CONFIG = {
    'ig': {
        'label': 'Integrated Gradients',
        'local_path': DEFAULT_VIS_DIR / 'tfidf_integrated_gradients_local.csv',
        'global_path': DEFAULT_VIS_DIR / 'tfidf_integrated_gradients_global.csv',
        'score_key': 'attribution',
        'global_score_key': 'mean_attribution',
    },
    'lrp': {
        'label': 'LRP',
        'local_path': DEFAULT_VIS_DIR / 'tfidf_lrp_local.csv',
        'global_path': DEFAULT_VIS_DIR / 'tfidf_lrp_global.csv',
        'score_key': 'relevance',
        'global_score_key': 'mean_relevance',
    },
}


def set_csv_field_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create HTML text highlights for TF-IDF IG/LRP explanation terms.'
    )
    parser.add_argument(
        '--method',
        choices=('ig', 'lrp', 'both'),
        default='both',
        help='Explanation method to visualize.',
    )
    parser.add_argument(
        '--scope',
        choices=('local', 'global'),
        default='local',
        help='Use document-local terms or class-global terms.',
    )
    parser.add_argument(
        '--row_index',
        '--row-index',
        dest='row_index',
        type=int,
        default=0,
        help='Row index in the local explanation CSV used to select the document.',
    )
    parser.add_argument(
        '--file_path',
        '--file-path',
        dest='file_path',
        type=Path,
        default=None,
        help='Optional explicit text file to highlight. If omitted, row_index selects one.',
    )
    parser.add_argument(
        '--class_label',
        '--class-label',
        dest='class_label',
        choices=('homme', 'femme'),
        default=None,
        help='Class terms to use for global explanations. Defaults to the selected row explained class.',
    )
    parser.add_argument(
        '--n_terms',
        '--n-terms',
        dest='n_terms',
        type=int,
        default=20,
        help='Number of positive and negative terms to show.',
    )
    parser.add_argument(
        '--side',
        choices=('positive', 'negative', 'both'),
        default='both',
        help='For local explanations, choose positive, negative, or both terms.',
    )
    parser.add_argument(
        '--max_chars',
        '--max-chars',
        dest='max_chars',
        type=int,
        default=20000,
        help='Maximum number of text characters to include in the HTML.',
    )
    parser.add_argument(
        '--output_dir',
        '--output-dir',
        dest='output_dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Directory where HTML files are written.',
    )
    parser.add_argument(
        '--output_html',
        '--output-html',
        dest='output_html',
        type=Path,
        default=None,
        help='Optional explicit output path. Only valid when method is ig or lrp.',
    )
    parser.add_argument(
        '--ig_local_path',
        '--ig-local-path',
        dest='ig_local_path',
        type=Path,
        default=METHOD_CONFIG['ig']['local_path'],
        help='Path to local Integrated Gradients CSV.',
    )
    parser.add_argument(
        '--ig_global_path',
        '--ig-global-path',
        dest='ig_global_path',
        type=Path,
        default=METHOD_CONFIG['ig']['global_path'],
        help='Path to global Integrated Gradients CSV.',
    )
    parser.add_argument(
        '--lrp_local_path',
        '--lrp-local-path',
        dest='lrp_local_path',
        type=Path,
        default=METHOD_CONFIG['lrp']['local_path'],
        help='Path to local LRP CSV.',
    )
    parser.add_argument(
        '--lrp_global_path',
        '--lrp-global-path',
        dest='lrp_global_path',
        type=Path,
        default=METHOD_CONFIG['lrp']['global_path'],
        help='Path to global LRP CSV.',
    )
    return parser.parse_args()


def method_paths(args, method):
    if method == 'ig':
        return args.ig_local_path, args.ig_global_path
    return args.lrp_local_path, args.lrp_global_path


def read_csv_rows(path):
    set_csv_field_limit()
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def resolve_text_path(path_value):
    candidates = [
        Path(path_value),
        WORKSPACE_ROOT / Path(path_value),
        TFIDF_ROOT / Path(path_value),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Could not resolve text path: {path_value}')


def load_text(path, max_chars):
    text = Path(path).read_text(encoding='utf-8', errors='ignore')
    if max_chars and max_chars > 0 and len(text) > max_chars:
        return text[:max_chars], True
    return text, False


def select_local_row(local_rows, row_index, file_path):
    if file_path is not None:
        wanted = normalize_path_for_match(file_path)
        for row in local_rows:
            if normalize_path_for_match(row['file_path']) == wanted:
                return row
        raise ValueError(f'No local explanation row found for file_path={file_path}')

    if row_index < 0 or row_index >= len(local_rows):
        raise IndexError(f'row_index={row_index} outside local CSV range 0..{len(local_rows) - 1}')
    return local_rows[row_index]


def normalize_path_for_match(path_value):
    return str(path_value).replace('\\', '/').lower()


def parse_term_records(raw_json, score_key, direction, n_terms):
    if not raw_json:
        return []

    records = json.loads(raw_json)
    parsed = []
    for rank, record in enumerate(records[:n_terms], start=1):
        term = str(record.get('term', '')).strip()
        if not term:
            continue
        score = float(record.get(score_key, 0.0))
        parsed.append({
            'term': term,
            'score': score,
            'tfidf': float(record.get('tfidf', 0.0)),
            'rank': rank,
            'direction': direction,
        })
    return parsed


def local_terms(row, method, n_terms, side):
    score_key = METHOD_CONFIG[method]['score_key']
    terms = []

    if side in ('positive', 'both'):
        terms.extend(parse_term_records(row.get('top_positive_terms'), score_key, 'positive', n_terms))
    if side in ('negative', 'both'):
        terms.extend(parse_term_records(row.get('top_negative_terms'), score_key, 'negative', n_terms))

    return terms


def global_terms(global_rows, method, class_label, n_terms):
    score_key = METHOD_CONFIG[method]['global_score_key']
    terms = []

    class_rows = [row for row in global_rows if row.get('class_label') == class_label]
    class_rows.sort(key=lambda row: int(row.get('rank', 10**9)))

    for row in class_rows[:n_terms]:
        term = str(row.get('term', '')).strip()
        if not term:
            continue
        terms.append({
            'term': term,
            'score': float(row.get(score_key, 0.0)),
            'tfidf': None,
            'rank': int(row.get('rank', len(terms) + 1)),
            'direction': 'positive',
        })

    return terms


def term_pattern(term):
    escaped = re.escape(term)
    return re.compile(rf'(?<!\w){escaped}(?!\w)', re.IGNORECASE)


def find_highlight_spans(text, terms):
    candidates = []
    for term in terms:
        for match in term_pattern(term['term']).finditer(text):
            candidates.append({
                'start': match.start(),
                'end': match.end(),
                'term': term,
                'priority': (abs(term['score']), match.end() - match.start()),
            })

    candidates.sort(key=lambda item: (-item['priority'][0], -item['priority'][1], item['start']))
    selected = []
    occupied = []

    for candidate in candidates:
        start = candidate['start']
        end = candidate['end']
        if any(start < used_end and end > used_start for used_start, used_end in occupied):
            continue
        selected.append(candidate)
        occupied.append((start, end))

    selected.sort(key=lambda item: item['start'])
    return selected


def rgba_for_score(score, max_abs_score, direction):
    if max_abs_score <= 0:
        alpha = 0.25
    else:
        alpha = 0.18 + 0.45 * min(abs(score) / max_abs_score, 1.0)

    if direction == 'negative':
        return f'rgba(214, 91, 91, {alpha:.3f})'
    return f'rgba(73, 157, 112, {alpha:.3f})'


def render_highlighted_text(text, spans, max_abs_score):
    chunks = []
    cursor = 0

    for span in spans:
        start = span['start']
        end = span['end']
        term = span['term']
        direction = term['direction']
        background = rgba_for_score(term['score'], max_abs_score, direction)
        title = (
            f"{term['term']} | {direction} | score={term['score']:.6g} | "
            f"rank={term['rank']}"
        )
        chunks.append(html.escape(text[cursor:start]))
        chunks.append(
            '<mark class="hit {direction}" style="background:{background}" title="{title}">'
            '{content}</mark>'.format(
                direction=html.escape(direction),
                background=background,
                title=html.escape(title, quote=True),
                content=html.escape(text[start:end]),
            )
        )
        cursor = end

    chunks.append(html.escape(text[cursor:]))
    return ''.join(chunks)


def occurrence_counts(spans):
    counts = {}
    for span in spans:
        key = span['term']['term']
        counts[key] = counts.get(key, 0) + 1
    return counts


def render_terms_table(terms, spans):
    counts = occurrence_counts(spans)
    rows = []
    terms_sorted = sorted(terms, key=lambda item: (item['direction'] != 'positive', item['rank']))

    for term in terms_sorted:
        tfidf_text = '' if term['tfidf'] is None else f'{term["tfidf"]:.6g}'
        rows.append(
            '<tr class="{direction}">'
            '<td>{rank}</td><td>{direction}</td><td>{term}</td><td>{score:.6g}</td>'
            '<td>{tfidf}</td><td>{count}</td></tr>'.format(
                direction=html.escape(term['direction']),
                rank=int(term['rank']),
                term=html.escape(term['term']),
                score=float(term['score']),
                tfidf=html.escape(tfidf_text),
                count=counts.get(term['term'], 0),
            )
        )

    return '\n'.join(rows)


def output_path_for(args, method, scope, row_index, class_label):
    if args.output_html is not None:
        if args.method == 'both':
            raise ValueError('--output_html can only be used with --method ig or --method lrp')
        return args.output_html

    slug = (
        f'tfidf_text_highlight_method-{method}_scope-{scope}'
        f'_row-{row_index}_class-{class_label}_terms-{args.n_terms}.html'
    )
    return args.output_dir / slug


def render_html(
    method,
    scope,
    row,
    class_label,
    text_path,
    text,
    was_truncated,
    terms,
    spans,
):
    method_label = METHOD_CONFIG[method]['label']
    max_abs_score = max([abs(term['score']) for term in terms] or [0.0])
    highlighted_text = render_highlighted_text(text, spans, max_abs_score)
    terms_table = render_terms_table(terms, spans)

    subtitle_parts = [
        f'Method: {method_label}',
        f'Scope: {scope}',
        f'Class terms: {class_label}',
    ]
    if row:
        subtitle_parts.extend([
            f'True label: {row.get("true_label", "")}',
            f'Prediction: {row.get("predicted_label", "")}',
            f'Confidence: {float(row.get("confidence", 0.0)):.4f}',
        ])
    subtitle = ' | '.join(subtitle_parts)
    truncated_note = (
        '<p class="warning">Text was truncated by --max_chars for readability.</p>'
        if was_truncated else ''
    )

    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TF-IDF {html.escape(method_label)} Text Highlight</title>
  <style>
    :root {{
      color-scheme: light;
      --border: #d8dee4;
      --text: #1f2328;
      --muted: #667085;
      --positive: #2f7d4f;
      --negative: #b42318;
      --surface: #f6f8fa;
    }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--text);
      background: #ffffff;
    }}
    header {{
      padding: 22px 28px 16px;
      border-bottom: 1px solid var(--border);
      background: var(--surface);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .subtitle, .path, .warning {{
      margin: 4px 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
      gap: 24px;
      padding: 24px 28px;
      align-items: start;
    }}
    .text {{
      max-width: 980px;
      white-space: pre-wrap;
      line-height: 1.72;
      font-size: 16px;
    }}
    .hit {{
      border-radius: 3px;
      padding: 1px 3px;
      box-decoration-break: clone;
      -webkit-box-decoration-break: clone;
      cursor: help;
    }}
    .hit.positive {{
      border-bottom: 2px solid var(--positive);
    }}
    .hit.negative {{
      border-bottom: 2px solid var(--negative);
    }}
    aside {{
      position: sticky;
      top: 16px;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }}
    aside h2 {{
      margin: 0;
      padding: 14px 16px;
      font-size: 16px;
      border-bottom: 1px solid var(--border);
      background: var(--surface);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      background: #fbfbfc;
    }}
    tr.positive td:nth-child(2) {{
      color: var(--positive);
      font-weight: 700;
    }}
    tr.negative td:nth-child(2) {{
      color: var(--negative);
      font-weight: 700;
    }}
    @media (max-width: 980px) {{
      main {{
        grid-template-columns: 1fr;
      }}
      aside {{
        position: static;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>TF-IDF Text Highlight</h1>
    <p class="subtitle">{html.escape(subtitle)}</p>
    <p class="path">{html.escape(str(text_path))}</p>
    {truncated_note}
  </header>
  <main>
    <article class="text">{highlighted_text}</article>
    <aside>
      <h2>Highlighted Terms</h2>
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Side</th>
            <th>Term</th>
            <th>Score</th>
            <th>TF-IDF</th>
            <th>Hits</th>
          </tr>
        </thead>
        <tbody>
          {terms_table}
        </tbody>
      </table>
    </aside>
  </main>
</body>
</html>
'''


def generate_for_method(args, method):
    local_path, global_path = method_paths(args, method)
    local_rows = read_csv_rows(local_path)
    row = select_local_row(local_rows, args.row_index, args.file_path)
    text_path = resolve_text_path(args.file_path or row['file_path'])

    if args.scope == 'local':
        class_label = row.get('explained_class') or row.get('predicted_label') or row.get('true_label')
        terms = local_terms(row, method, args.n_terms, args.side)
    else:
        class_label = args.class_label or row.get('explained_class') or row.get('predicted_label')
        global_rows = read_csv_rows(global_path)
        terms = global_terms(global_rows, method, class_label, args.n_terms)

    if not terms:
        raise ValueError(f'No terms found for method={method}, scope={args.scope}, class={class_label}')

    text, was_truncated = load_text(text_path, args.max_chars)
    spans = find_highlight_spans(text, terms)
    output_path = output_path_for(args, method, args.scope, args.row_index, class_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        render_html(
            method=method,
            scope=args.scope,
            row=row,
            class_label=class_label,
            text_path=text_path,
            text=text,
            was_truncated=was_truncated,
            terms=terms,
            spans=spans,
        ),
        encoding='utf-8',
    )
    print(f'Saved {METHOD_CONFIG[method]["label"]} text highlight: {output_path}')
    print(f'Highlighted occurrences: {len(spans)}')


def main():
    args = parse_args()
    methods = ('ig', 'lrp') if args.method == 'both' else (args.method,)

    for method in methods:
        generate_for_method(args, method)


if __name__ == '__main__':
    main()

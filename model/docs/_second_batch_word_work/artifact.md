# Template execution contract

## Reference

- Source: `C:\Users\JOHN MARK\Code\ai-research-main\model\docs\Pneumonia_Model_Training_Details_Black_Research_Complete_20260802.docx`
- Read-only working copy: `C:\Users\JOHN MARK\Code\ai-research-main\model\docs\_second_batch_word_work\reference_copy.docx`
- Working-copy SHA-256: `96F95CB0856B5E21A0CED843E6FF8B279BD5E80C3946C5E1964082520C7CAC0C`
- Reference size: 4,532,879 bytes at discovery; the saved copy is authoritative while the source is open in Word.
- Word structural count: 33 pages, 1 section, 1,917 paragraphs, 35 tables, and 9 inline images.
- Render status: the packaged renderer cannot run because LibreOffice is absent. Microsoft Word opens the copy and reports 33 pages, but PDF export of the open-source-derived copy did not complete within five minutes. The final appended section must therefore be rendered independently before merge, and the merged DOCX must receive structural and image audits.

## Page system

- US Letter portrait, 8.5 x 11 inches.
- Margins: 1.0 inch on every side.
- Header/footer distance: approximately 0.492 inch.
- One section; NEW_PAGE start; different first-page header/footer enabled.
- The append operation may increase page count but must not create a new page geometry or alter the original section properties.

## Typography and color

- Arial throughout; black text and black/gray borders only.
- Normal: 10.5 pt, 1.1 line spacing, 0 pt before, 6 pt after.
- Title: 28 pt bold, 8 pt after.
- Heading 1: 16 pt bold, 16 pt before, 8 pt after.
- Heading 2: 13 pt bold, 12 pt before, 6 pt after.
- Heading 3: 11.5 pt bold, 8 pt before, 4 pt after.
- Caption: 8.5 pt, single spacing, 0 pt after.
- The user explicitly requested no blue decoration. Scientific pixels may retain their algorithmically generated values only when necessary; the document visual system remains black and grayscale.

## Lists, tables, and figures

- Reuse the reference's real Heading 1/2/3, Normal, Caption, and List Bullet styles.
- Reference tables use the `Normal Table` style. New result tables must use explicit fixed widths totaling 9,120 DXA inside the 9,360-DXA usable width, a 120-DXA table indent, repeating header rows, black borders, light-gray header fill, black text, and non-fixed row heights.
- Numeric result columns are centered; model and condition columns are left aligned.
- Figures must be inline, centered, sized no wider than 6.2 inches, followed immediately by a Caption paragraph, and must include descriptive alternative text.

## Components and content flow

- Preserve all existing title, executive summary, Sections 1-10, headers, footers, tables, figures, citations, and final audit language.
- Append a page break and a new Heading 1 section titled `11. Second-Batch Training: Shared Data Flow and Complete Results`.
- Within the appended section, place content in this order:
  1. Shared-versus-variable explanation.
  2. One-image data-preparation flow and definitions.
  3. Same-image preprocessing comparison.
  4. Same-image training-only augmentation flow and definitions.
  5. End-to-end training/validation pipeline and definitions.
  6. Main architecture/preprocessing results table.
  7. Tuning-trial table.
  8. Seed-confirmation table and model-selection explanation.

## Editable slot map

- `word/document.xml`, after the existing final paragraph: append-only slot for the new section.
- New image relationships/media parts: permitted for the four new flow figures.
- Existing paragraphs, tables, drawings, relationships, headers, footers, styles, numbering, and media: preserve semantically and visually; no intentional edits.

## Package preservation

- Preserve the existing section properties, styles, numbering, headers/footers, nine existing image relationships, and all existing body content.
- `python-docx` is permitted because the requested append requires new paragraphs, tables, images, relationships, and media. The original reference remains unchanged.
- No comments, tracked changes, content controls, footnotes, endnotes, or field refreshes are required for the appended section.

## Fidelity gates

- The source path remains untouched and open-user state is not altered.
- The new section uses the existing black Arial hierarchy.
- One actual manifest-listed training image is used consistently across every visual example.
- Flow figures are legible at 100% and do not overlap captions or margins.
- Tables repeat headers, wrap safely, and contain all 21 main runs, 12 tuning trials, and 3 seed-confirmation runs without mixing smoke tests into model-selection evidence.
- Structural audits after merge must confirm one section, portrait Letter geometry, preserved existing heading counts plus the new headings, and 13 inline images total.

-- Basic size checks
SELECT COUNT(*) AS works FROM works;
SELECT COUNT(*) AS authors FROM authors;
SELECT COUNT(*) AS pdf_rows FROM pdf_downloads;

-- Papers by year
SELECT publication_year, COUNT(*) AS papers
FROM works
GROUP BY publication_year
ORDER BY publication_year DESC;

-- Most cited papers in the local corpus
SELECT work_id, title, publication_year, cited_by_count, primary_source_name
FROM works
ORDER BY cited_by_count DESC
LIMIT 20;

-- Current PDF download status
SELECT download_status, COUNT(*) AS n
FROM download_queue
GROUP BY download_status
ORDER BY n DESC;

-- Downloaded PDFs with local paths
SELECT work_id, title, publication_date, file_path, file_size_bytes
FROM download_queue
WHERE download_status = 'success'
ORDER BY publication_date DESC
LIMIT 20;

-- Most common topics in the local corpus
SELECT t.display_name, COUNT(*) AS papers
FROM work_topics wt
JOIN topics t ON t.topic_id = wt.topic_id
GROUP BY t.topic_id, t.display_name
ORDER BY papers DESC
LIMIT 20;

-- Most prolific authors in the local corpus
SELECT a.display_name, COUNT(*) AS papers
FROM work_authorships wa
JOIN authors a ON a.author_id = wa.author_id
GROUP BY a.author_id, a.display_name
ORDER BY papers DESC
LIMIT 20;

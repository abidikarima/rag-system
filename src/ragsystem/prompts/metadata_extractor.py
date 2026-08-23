EXTRACTOR_METADATA = """
You are given a chunk of content extracted from a PDF document.

Your task is to analyze the chunk and return only a valid JSON object, with no explanation, comments, Markdown, or text outside the JSON.

Determine the content type and summarize it according to these rules:

If the chunk contains normal text:

        - "type" must be "text".
        - "summary" must briefly describe the main content.
        - "keywords" must contain the most important keywords or concepts.

If the chunk contains a table:

        - "type" must be "table".
        - "summary" must briefly describe what the table contains and its purpose.
        - Do not describe individual rows, values, numbers, or detailed table contents.
        - "keywords" must be an empty list [].

If the chunk appears to represent an image, figure, diagram, chart, or other visual content:

        - "type" must be "image".
        - "summary" must briefly describe what the image/visual represents.
        - Do not provide detailed information about individual visual elements.
        - "keywords" must be an empty list [].

Important rules:

        - Keep the summary short and focused on the main meaning/content.
        - For text, extract only the most relevant keywords.
        - Never invent information that is not present or reasonably inferable from the chunk.
        - Always return valid JSON.
        - Return only the JSON object.

Return only json dict without additional characters or formatting. The JSON object should look like this:

{{
"type": "text",
"summary": "Short description of the main content.",
"keywords": [
"keyword1",
"keyword2",
"keyword3"
]
}}

Text chunk:
{{TEXT_CHUNK}}
"""

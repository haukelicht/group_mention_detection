import re

def glob_to_regex(glob):
    """
    Convert a glob pattern to a regular expression.
    """
    # Escape all characters that are special in regex, except for *, ?, and []
    regex = re.escape(glob)
    
    # Replace the escaped glob wildcards with regex equivalents
    regex = regex.replace(r'\*', '\S*?')
    regex = regex.replace(r'\?', '.')
    regex = regex.replace(r'\[', '[')
    regex = regex.replace(r'\]', ']')
    
    # Add anchors to match the entire string
    regex = r'\b' + regex + r'\b'
    
    return regex

def apply_keywords(text, keywords):
    out = []
    for c, kws in keywords.items():
        for kw in kws: 
            for m in re.finditer(re.compile(kw, re.IGNORECASE), text):
                out.append([c, kw, m.span(), m.group()])
    return out
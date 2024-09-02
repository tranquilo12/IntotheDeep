#!/usr/bin/env fish

# Read .gitignore contents
set gitignore_file ".gitignore"
if test -f $gitignore_file
    set gitignore_contents (cat $gitignore_file | string trim)
else
    echo "No .gitignore file found. Only *.pyc files will be ignored."
    set gitignore_contents
end

# Process gitignore patterns
set ignore_patterns
for pattern in $gitignore_contents
    # Skip empty lines and comments
    if test -n "$pattern" 
        and test (string sub -l 1 "$pattern") != "#"
        # Remove trailing slash if present
        set pattern (string replace -r '/$' '' -- $pattern)
        set ignore_patterns $ignore_patterns $pattern
    end
end

# Add *.pyc to ignore patterns
set ignore_patterns $ignore_patterns "*.pyc"

# Join patterns with pipe for tree command
set ignore_string (string join '|' $ignore_patterns)


# Use tree command with ignore patterns
tree -I "$ignore_string" .


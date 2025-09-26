
import re
import textwrap


# used to parse tasks and prompts from markdown files
def parse_markdown_to_dict(markdown_text):
    # Split the markdown into lines
    lines = markdown_text.split('\n')
    
    # First pass: identify all sections with their levels, content, and code blocks
    sections = []
    current_section = None
    current_content = []
    current_code = []
    in_code_block = False
    under_test_heading = False
    under_tools_heading = False
    
    for line in lines:
        # Check for headings
        heading_match = re.match(r'^(#{1,6})\s+(.*?)$', line)
        
        if heading_match:
            # Save the current section if there is one
            if current_section is not None:
                if (under_test_heading or under_tools_heading) and current_code:
                    # For Test sections, extract code separately
                    current_section["code"] = "\n".join(current_code).strip()
                    current_section["content"] = "\n".join(current_content).strip()
                else:
                    # For other sections, keep code blocks as part of content
                    current_section["content"] = "\n".join(current_content).strip()
                
                sections.append(current_section)
            
            # Start a new section
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            current_section = {"title": title, "level": level}
            current_content = []
            current_code = []
            in_code_block = False
            under_test_heading = (title == "Test") or (title == "Tools")  # Check if this is a Test or Tools heading
        else:
            if line.strip() == "```" or line.strip().startswith("```"):
                # Handle code block delimiter
                in_code_block = not in_code_block
                
                # Always include the code block delimiter in content for non-Test sections
                if not under_test_heading:
                    current_content.append(line)
                    
                # For Test sections, we skip the delimiter in both content and code
                continue
            
            # Add line to appropriate collection
            if in_code_block and under_test_heading:
                # Only extract code separately for Test headings
                current_code.append(line)
            else:
                # For everything else, add to content
                current_content.append(line)
    
    # Don't forget the last section
    if current_section is not None:
        if under_test_heading and current_code:
            # For Test sections, extract code separately
            current_section["code"] = "\n".join(current_code).strip()
            current_section["content"] = "\n".join(current_content).strip()
        else:
            # For other sections, keep code blocks as part of content
            current_section["content"] = "\n".join(current_content).strip()
            
        sections.append(current_section)
    
    # Second pass: build the nested dictionary structure
    def build_nested_dict(sections, start_idx=0, min_level=1):
        result = {}
        i = start_idx
        
        while i < len(sections):
            section = sections[i]
            
            # If we encounter a section with a level lower than what we're processing,
            # we've moved back up the hierarchy
            if section["level"] < min_level:
                return result, i
            
            # If this section is at the current level, add it to our result
            if section["level"] == min_level:
                node = {"content": section["content"]}
                
                # Add code if it exists (only for Test sections)
                if "code" in section:
                    node["code"] = section["code"]
                
                # Process children (sections with higher level)
                children, next_i = build_nested_dict(sections, i + 1, min_level + 1)
                
                # Only add children if there are any
                if children:
                    node["children"] = children
                
                result[section["title"]] = node
                i = next_i
            else:
                # Skip sections with higher level than what we're currently processing
                i += 1
        
        return result, i
    
    nested_dict, _ = build_nested_dict(sections)
    return nested_dict

# Helper function to print the dictionary nicely
def print_dict(d, indent=0):
    for key, value in d.items():
        print(' ' * indent + str(key) + ':')
        if isinstance(value, dict):
            if "content" in value:
                print(' ' * (indent + 4) + "content:")
                content_lines = value["content"].split('\n')
                for line in content_lines:
                    print(' ' * (indent + 8) + line)
                
                if "code" in value:
                    print(' ' * (indent + 4) + "code:")
                    code_lines = value["code"].split('\n')
                    for line in code_lines:
                        print(' ' * (indent + 8) + line)
                
                if "children" in value:
                    print(' ' * (indent + 4) + "children:")
                    print_dict(value["children"], indent + 8)
            else:
                print_dict(value, indent + 4)
        else:
            print(' ' * (indent + 4) + str(value))


def tools_parse_markdown_to_dict(markdown_text):
    # Split the markdown into lines
    lines = markdown_text.split('\n')
    
    # First pass: identify all sections with their levels, content, and code blocks
    sections = []
    current_section = None
    current_content = []
    current_code = []
    in_code_block = False
    
    for line in lines:
        # Check for headings
        heading_match = re.match(r'^(#{1,6})\s+(.*?)$', line)
        
        if heading_match:
            # Save the current section if there is one
            if current_section is not None:
                # Extract code separately if it exists (for all heading levels)
                if current_code:
                    current_section["code"] = "\n".join(current_code).strip()
                
                # Save content for all sections
                current_section["content"] = "\n".join(current_content).strip()
                sections.append(current_section)
            
            # Start a new section
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            current_section = {"title": title, "level": level}
            current_content = []
            current_code = []
            in_code_block = False
        else:
            # Check for code block delimiters
            if line.strip() == "```" or line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue  # Skip the delimiter line in both content and code
            
            # Add line to appropriate collection
            if in_code_block:
                # Extract code separately for all headings
                current_code.append(line)
            else:
                # For non-code lines, add to content
                current_content.append(line)
    
    # Don't forget the last section
    if current_section is not None:
        # Extract code separately if it exists
        if current_code:
            current_section["code"] = "\n".join(current_code).strip()
        
        # Save content for all sections
        current_section["content"] = "\n".join(current_content).strip()
        sections.append(current_section)
    
    # Second pass: build the nested dictionary structure
    def build_nested_dict(sections, start_idx=0, min_level=1):
        result = {}
        i = start_idx
        
        while i < len(sections):
            section = sections[i]
            
            # If we encounter a section with a level lower than what we're processing,
            # we've moved back up the hierarchy
            if section["level"] < min_level:
                return result, i
            
            # If this section is at the current level, add it to our result
            if section["level"] == min_level:
                node = {"content": section["content"]}
                
                # Add code if it exists (for all sections)
                if "code" in section:
                    node["code"] = section["code"]
                
                # Process children (sections with higher level)
                children, next_i = build_nested_dict(sections, i + 1, min_level + 1)
                
                # Only add children if there are any
                if children:
                    node["children"] = children
                
                result[section["title"]] = node
                i = next_i
            else:
                # Skip sections with higher level than what we're currently processing
                i += 1
        
        return result, i
    
    nested_dict, _ = build_nested_dict(sections)
    return nested_dict

# Helper function to print the dictionary nicely
def print_dict(d, indent=0):
    for key, value in d.items():
        print(' ' * indent + str(key) + ':')
        if isinstance(value, dict):
            if "content" in value:
                print(' ' * (indent + 4) + "content:")
                if value["content"]:
                    content_lines = value["content"].split('\n')
                    for line in content_lines:
                        print(' ' * (indent + 8) + line)
                else:
                    print(' ' * (indent + 8) + "(empty)")
                
                if "code" in value:
                    print(' ' * (indent + 4) + "code:")
                    code_lines = value["code"].split('\n')
                    for line in code_lines:
                        print(' ' * (indent + 8) + line)
                
                if "children" in value:
                    print(' ' * (indent + 4) + "children:")
                    print_dict(value["children"], indent + 8)
            else:
                print_dict(value, indent + 4)
        else:
            print(' ' * (indent + 4) + str(value))

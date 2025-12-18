import os

IGNORE_DIRS = {
    '.git', '__pycache__', 'venv', 'env', 
    '.idea', '.vscode', 'build', 'dist', 
    'egg-info', 'node_modules'
}

INCLUDE_EXTS = {'.py', '.md', '.toml', '.json', '.yaml', '.yml'}

def pack_project():
    output_file = 'project_context.txt'
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                _, ext = os.path.splitext(file)
                if ext in INCLUDE_EXTS:
                    file_path = os.path.join(root, file)
                    
                    if file == 'pack_project.py' or file == output_file:
                        continue

                    outfile.write(f"\n{'='*30}\n")
                    outfile.write(f"FILE: {file_path}\n")
                    outfile.write(f"{'='*30}\n\n")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"Error reading file: {e}\n")
                    
                    outfile.write("\n")
    
    print(f"Project packed into {output_file}")

if __name__ == "__main__":
    pack_project()
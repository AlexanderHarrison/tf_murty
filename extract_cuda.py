import sys

HEADER = "const char *cuda_program[] ="

if __name__ == "__main__":
    filename = sys.argv[1] + ".c"

    with open(filename) as f:
        file = f.read()
        header_start = file.index(HEADER) + len(HEADER)
        brace_start = file.index('{', header_start)
        
        i = brace_start + 1
        depth_count = 1
        while depth_count > 0:
            lbrace_i = file.index('{', i)
            rbrace_i = file.index('}', i)
            if lbrace_i < rbrace_i:
                depth_count += 1
                i = lbrace_i + 1
            else:
                depth_count -= 1
                i = rbrace_i + 1

        cuda_list = file[(brace_start+2):(i-8)]
        unlisted = cuda_list.replace('",\n            "', "")
        cuda = unlisted.replace('\\n', "\n")

        with open(sys.argv[1] + ".cu", "w") as out:
            out.write(cuda)

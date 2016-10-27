# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:42:43 2016

@author: JKupka
"""

latex_file_path = "..\\_build\\latex\\pandapower.tex"
version = "1.0"

class AlreadyProcessedException(Exception):
    pass

def postprocess_latex():
    latex_file = open(latex_file_path, 'r+')

    lines = latex_file.readlines()
    bufferlines = ['%postprocessed\n']

    # special place variables
    document_end = False
    tab_begin = False
    module_index = False
#    parameters = False
#    parameters_next = False

    for i, line in enumerate(lines):
        if line.find('%postprocessed') != -1:
            latex_file.close()
            raise AlreadyProcessedException('This file is already postprocessed! Please recompile it first!')


        # find special places in texfile
        if line.find('\\begin{tab') != -1:
            tab_begin = True
        if line.find('\\renewcommand{\indexname}') != -1:
            module_index = True
        if line.find('\\end{document}') != -1:
            document_end = True
#        if line.find('\\textbf{Parameters}') != -1: # or line.find('\\strong{See also') != -1:
#            parameters = True

        ############################################
        # replace commands and restructure texfile #
        ############################################

        # maketitle
        if line.find('\\maketitle') != -1:
            bufferlines.append('{\\let\\newpage\\relax\\maketitle}\n')
            bufferlines.append('\\subimport{../../report/}{author_page}')
            bufferlines.append('\\newpage')

#        #set title
        elif line.find('title{pandapower') != -1:
            bufferlines.append('\\title{\\Huge pandapower \\\\[1em]  {\\Large - Convenient Power System Modelling and Analysis \\\\ based on PYPOWER and pandas - }}\n')
            bufferlines.append('\\subtitle{TECHNICAL REPORT}\n')

        # change documentclass and use xcolor package
        elif line.find('\\documentclass') != -1:
            bufferlines.append('\\documentclass[a4paper, 10pt, titlepage]{article}\n')
            bufferlines.append('\\usepackage{xcolor}\n')
            bufferlines.append('\\usepackage{import}\n')
            bufferlines.append('\\usepackage{pdflscape}\n')
            bufferlines.append('\\usepackage{afterpage}\n')
#            bufferlines.append('\\usepackage{hyphenat}\n')
        # set tabulary to longtable and change pagestyle
        elif line.find('\\renewenvironment') != -1:
            bufferlines.append(line)
            bufferlines.append('\\renewenvironment{tabulary}{\\begin{longtable}}{\\end{longtable}}\n')
            bufferlines.append('\n\\pagestyle{headings}\n\n')
        #set tocdepth
        elif line.find('\setcounter{tocdepth}') != -1:
            bufferlines.append('\setcounter{tocdepth}{2}\n')
        # define endheader for longtables
        elif tab_begin and line.find('\\\\') != -1:
            bufferlines.append(line)
            bufferlines.append('\\endhead\n')
            tab_begin = False
        elif line.find('(\\autopageref') != -1:
            startindex = line.find('(\\autopageref')
            endindex = line.find(')', startindex)
            bufferlines.append(line[0:startindex-1] + line[endindex+1:len(line)])
        elif line.find('\\date{') != -1:
            bufferlines.append('\\date{\\today \\\\[.5em] Version %s}\n'%version)
        elif line.find("subsection{Lines}") != -1:
            bufferlines.append("\\thispagestyle{empty} \n")
            bufferlines.append("\\pagestyle{empty} \n")
            bufferlines.append("\\begin{landscape} \n")
            bufferlines.append("\\footnotesize \n")
            bufferlines.append(replace(line))
        elif line.find("{Manage Standard Types}") != -1:
            bufferlines.append("\\end{landscape} \n")
            bufferlines.append("\\restoregeometry \n")
            bufferlines.append("\\pagestyle{headings} \n")
            bufferlines.append("\\normal \n")
            bufferlines.append(replace(line))

            #        elif parameters:
#            parameters = False
#            parameters_next = True
#            bufferlines.append(line)
#        elif parameters_next:
#            if line == '\n':
#                continue
#            else:
#                parameters_next = False
#                bufferlines.append(line)
            
        # mask out module index at the document end
        elif module_index and not document_end:
            continue
        # change chapter to section etc, replace unnecessary commands
        else:
            bufferlines.append(replace(line))

    # write changes to file
    latex_file.seek(0)
    latex_file.truncate()
    for line in bufferlines:
        latex_file.write(line)

    latex_file.close()

def replace(line):
    line = line.replace('\\subsection', '\\subsubsection')
    line = line.replace('\\section', '\\subsection')
    line = line.replace('\\chapter', '\\clearpage\\section')
    line = line.replace('{\\linewidth}', '')
    line = line.replace('\\bigskip\\hrule{}\\bigskip', '\\bigskip')
    line = line.replace('{\\color{red}\\bfseries{}*}', '*')
    return line
    
postprocess_latex()

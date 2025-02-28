def prj_import(com_import, dst_user_dir, file_name):
    com_import.g_target = dst_user_dir
    com_import.g_file = file_name
    com_import.g_target = dst_user_dir
    assert com_import.g_target is dst_user_dir
    com_import.Execute()


def prj_dgs_import(com_import, dst_user_dir, file_name, name, template=None):
    com_import.targpath = dst_user_dir
    com_import.targname = name
    com_import.fFile = file_name
    if template is not None:
        com_import.prjTemplate = template
    com_import.Execute()


def choose_imp_dir(user, IMPFOLD):
    imp_dir_list = user.GetContents(IMPFOLD)
    if len(imp_dir_list) != 0:
        imp_dir = imp_dir_list[0]
    else:
        imp_dir = user.CreateObject('IntFolder', IMPFOLD)
    return imp_dir


def clear_dir(dir):
    trash = dir.GetContents()
    for item in trash:
        item.Delete()


def run_ldf(com_ldf):
    com_ldf.SetAttribute('iopt_net', 0)
    # com_ldf.SetAttribute('iopt_at', 1)
    com_ldf.SetAttribute('iopt_pq', 0)

    com_ldf.SetAttribute('errlf', 0.001)
    com_ldf.SetAttribute('erreq', 0.01)

    com_ldf.Execute()

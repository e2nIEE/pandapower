# make it faster


def echo_off(app, err=0, warn=0, info=0):
    ComEcho = app.GetFromStudyCase("ComEcho")
    ComEcho.iopt_updm = 0
    ComEcho.iopt_err = err
    ComEcho.iopt_wrng = warn
    ComEcho.iopt_info = info
    ComEcho.iopt_pcl = 0
    ComEcho.iopt_dpl = 0
    ComEcho.Execute()
    app.SetGraphicUpdate(0)


def echo_on(app):
    ComEcho = app.GetFromStudyCase("ComEcho")
    ComEcho.iopt_updm = 1
    ComEcho.iopt_err = 1
    ComEcho.iopt_wrng = 1
    ComEcho.iopt_info = 1
    ComEcho.iopt_pcl = 1
    ComEcho.iopt_dpl = 1
    ComEcho.Execute()
    app.SetGraphicUpdate(1)

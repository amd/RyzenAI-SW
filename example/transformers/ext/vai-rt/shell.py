import shlex


def run_shell(args, dry_run):
    print("running: " + " ".join([shlex.quote(arg) for arg in args]))
    if not dry_run:
        subprocess.check_call(args)

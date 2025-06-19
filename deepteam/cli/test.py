import time
import typer
import os
import sys
from typing import Optional
from typing_extensions import Annotated

app = typer.Typer(name="test")


def check_if_valid_file(test_file_or_directory: str):
    """Check if the provided path is a valid test file or directory."""
    if "::" in test_file_or_directory:
        test_file_or_directory, test_case = test_file_or_directory.split("::")
    if os.path.isfile(test_file_or_directory):
        if test_file_or_directory.endswith(".py"):
            if not os.path.basename(test_file_or_directory).startswith("test_"):
                raise ValueError(
                    "Test will not run. Please ensure the file starts with `test_` prefix."
                )
    elif os.path.isdir(test_file_or_directory):
        return
    else:
        raise ValueError(
            "Provided path is neither a valid file nor a directory."
        )


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def run(
    ctx: typer.Context,
    test_file_or_directory: str,
    color: str = "yes",
    durations: int = 10,
    pdb: bool = False,
    exit_on_first_failure: Annotated[
        bool, typer.Option("--exit-on-first-failure", "-x/-X")
    ] = False,
    show_warnings: Annotated[
        bool, typer.Option("--show-warnings", "-w/-W")
    ] = False,
    identifier: Optional[str] = typer.Option(
        None,
        "--identifier",
        "-id",
        help="Identify this test run with pytest",
    ),
    num_processes: Optional[int] = typer.Option(
        None,
        "--num-processes",
        "-n",
        help="Number of processes to use with pytest",
    ),
    repeat: Optional[int] = typer.Option(
        None,
        "--repeat",
        "-r",
        help="Number of times to rerun a test case",
    ),
    use_cache: Optional[bool] = typer.Option(
        False,
        "--use-cache",
        "-c",
        help="Whether to use cached results or not",
    ),
    ignore_errors: Optional[bool] = typer.Option(
        False,
        "--ignore-errors",
        "-i",
        help="Whether to ignore errors or not",
    ),
    skip_on_missing_params: Optional[bool] = typer.Option(
        False,
        "--skip-on-missing-params",
        "-s",
        help="Whether to skip test cases with missing parameters",
    ),
    verbose: Optional[bool] = typer.Option(
        None,
        "--verbose",
        "-v",
        help="Whether to turn on verbose mode for red teaming or not",
    ),
    mark: Optional[str] = typer.Option(
        None,
        "--mark",
        "-m",
        help="List of marks to run the tests with.",
    ),
):
    """Run red teaming tests using pytest integration."""
    check_if_valid_file(test_file_or_directory)
    
    # Import pytest here to avoid import errors if pytest is not installed
    try:
        import pytest
    except ImportError:
        typer.echo("Error: pytest is not installed. Please install it with: pip install pytest", err=True)
        raise typer.Exit(code=1)

    pytest_args = [test_file_or_directory]

    if exit_on_first_failure:
        pytest_args.insert(0, "-x")

    pytest_args.extend([
        "--verbose" if verbose else "--quiet",
        f"--color={color}",
        f"--durations={durations}",
        "-s",
    ])

    if pdb:
        pytest_args.append("--pdb")
    if not show_warnings:
        pytest_args.append("--disable-warnings")
    if num_processes is not None:
        pytest_args.extend(["-n", str(num_processes)])

    if repeat is not None:
        pytest_args.extend(["--count", str(repeat)])
        if repeat < 1:
            raise ValueError("The repeat argument must be at least 1.")

    if mark:
        pytest_args.extend(["-m", mark])
    if identifier:
        pytest_args.extend(["--identifier", identifier])

    # Append the extra arguments collected by allow_extra_args=True
    if ctx.args:
        pytest_args.extend(ctx.args)

    start_time = time.perf_counter()
    pytest_retcode = pytest.main(pytest_args)
    end_time = time.perf_counter()
    run_duration = end_time - start_time
    
    typer.echo(f"\nRed teaming test run completed in {run_duration:.2f} seconds")

    if pytest_retcode == 1:
        sys.exit(1)

    return pytest_retcode 
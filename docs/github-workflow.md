# Local development, GitHub, and the hosted app

The source repository is [paerade/nfl_ratings](https://github.com/paerade/nfl_ratings).
The hosted app is [nflratings.streamlit.app](https://nflratings.streamlit.app/).
The ranking app starts at root `app.py`; dependencies are in `requirements.txt`.
The release branch is `main`. Changes on a feature branch do not update the live
app until merged into the branch configured in Streamlit Community Cloud.

## Local setup

For a new checkout, clone the repository, create a Python 3.12 environment, and
install `requirements.txt`. Start the app from the repository root with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The first season load needs internet access to nflverse through nflreadpy.
No API key, local data archive, or research output is required. The app caches
prepared data for one hour. Refresh requests a new upstream snapshot; it cannot
create statistics that the provider has not published yet.

## Updating the code

1. Inspect `git status` before changing branches or pulling. Preserve local
   changes first; do not discard them to make a pull succeed.
2. Update `main` with `git pull --ff-only` when its working tree is ready.
3. Create a feature branch, make changes, and run `python -m pytest -q` after
   installing pytest. Review the diff and stage only files intended for release.
4. Commit, push the feature branch, and open a pull request targeting `main`.
5. Review and merge the pull request when ready to update the hosted app.
   Verify the deployment in Streamlit's management interface and open the app
   to confirm the new behavior.

Only committed and pushed files reach GitHub. Local virtual environments,
downloaded caches, secrets, and unrelated research should not be uploaded.
Adding a remote does not automatically commit or upload local work.

If the hosted app stays unchanged after a merge, verify its configured
repository, branch, and `app.py` entrypoint in Streamlit Community Cloud, then
inspect deployment logs. See Streamlit's
[deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
and [app management guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

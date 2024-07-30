W=${W:-$HOME/workspace}

echo "sync vaip"
(cd $W/vaip; git fetch --all)
(cd $W/vai-rt; git fetch --all;)

old_commit_id=$(cd $W/vai-rt; git show origin/dev:release_file/latest_dev.txt | grep vaip | awk '{print $2}' | sed 's/\r//g')
echo "old commit id = $old_commit_id"

new_commit_id=$(cd $W/vaip; git rev-parse origin/dev)
echo "new commit id = $new_commit_id"

if [ x"$old_commit_id" == x"$new_commit_id" ]; then
    echo "no update"
    exit 0
fi

cd $W/vai-rt;

branch_name="br_update_vaip_from_${old_commit_id:0:8}"
echo "branch_name = $branch_name"
if  git rev-parse --verify "$branch_name" >/dev/null 2>&1; then
    echo "Branch $branch_name exists."
else
    git branch $branch_name
    git reset --hard origin/dev
fi
git checkout --force $branch_name
git show origin/dev:release_file/latest_dev.txt |
    sed "s/vaip:.*/vaip: $new_commit_id/g" > release_file/latest_dev.txt;
git add release_file/latest_dev.txt;
title="update vaip from ${old_commit_id:0:8} to ${new_commit_id:0:8}"
body=$(
    cd ../vaip; echo "ChangeLog";
    env PAGER=cat git log --date=short --reverse --pretty=format:"- %h %s (by %an @ %ad)" $old_commit_id..$new_commit_id --date-order |
        sed 's:#\([0-9]*\):VitisAI/vaip#\1:g'
    )
msg=$(echo $title
      echo
      echo $body)

git commit -m "$msg"
git push --force -u fork $branch_name

title="update vaip from ${old_commit_id:0:8} to ${new_commit_id:0:8}"
whoami=$(gh --hostname gitenterprise.xilinx.com api  user --jq .login)
gh pr create --base dev --head $whoami:$branch_name --title "$title" --body "$body" || true
pr_number=$(gh pr list --head $branch_name --json number,title --jq '.[0].number')
if [ -z $pr_number ]; then
    echo "cannot find pr number"
    set -x
    gh pr list --head $branch_name --json number,title --jq '.[0].number'
    exit 1
fi
echo "pr_number = $pr_number"
gh pr edit $pr_number --title "$title" --body "$body"

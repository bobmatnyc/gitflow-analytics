#!/bin/bash
# Setup script to clone recess-recreo repositories

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Create repos directory
REPO_DIR="$HOME/repos/recess-recreo"
mkdir -p "$REPO_DIR"

echo "🚀 Setting up recess-recreo repositories..."
echo "   Repository directory: $REPO_DIR"
echo ""

# List of repositories
REPOS=(
    "frontend"
    "service-ts"
    "services"
    "scraper"
    "keycloak"
)

# Clone or update each repository
for repo in "${REPOS[@]}"; do
    REPO_PATH="$REPO_DIR/$repo"
    
    if [ -d "$REPO_PATH/.git" ]; then
        echo "📦 Updating $repo..."
        cd "$REPO_PATH"
        git pull origin main
    else
        echo "📥 Cloning $repo..."
        git clone "https://github.com/recess-recreo/$repo.git" "$REPO_PATH"
    fi
done

echo ""
echo "✅ Repository setup complete!"
echo ""
echo "You can now run:"
echo "  gitflow-analytics -c config-sample.yaml"
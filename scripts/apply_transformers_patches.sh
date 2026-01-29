#!/bin/bash
# Apply transformers patches required for PI0.5 PyTorch implementation

set -e

REQUIRED_VERSION="4.53.2"

echo "Applying transformers patches for PI0.5..."

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Get site-packages path
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

# Check if transformers is installed
if [ ! -d "$SITE_PACKAGES/transformers" ]; then
    echo "Error: transformers package not found at $SITE_PACKAGES/transformers"
    echo ""
    read -p "Would you like to install transformers==$REQUIRED_VERSION? [y/N] " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Installing transformers==$REQUIRED_VERSION..."
        pip install transformers==$REQUIRED_VERSION
    else
        echo "Aborted. Please install transformers==$REQUIRED_VERSION manually:"
        echo "   pip install transformers==$REQUIRED_VERSION"
        exit 1
    fi
fi

# Check transformers version
CURRENT_VERSION=$(python3 -c "import transformers; print(transformers.__version__)")
echo "   Current transformers version: $CURRENT_VERSION"
echo "   Required version: $REQUIRED_VERSION"

if [ "$CURRENT_VERSION" != "$REQUIRED_VERSION" ]; then
    echo ""
    echo "WARNING: Version mismatch detected!"
    echo "   Current: $CURRENT_VERSION"
    echo "   Required: $REQUIRED_VERSION"
    echo ""
    read -p "Would you like to reinstall transformers==$REQUIRED_VERSION? [y/N] " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Reinstalling transformers==$REQUIRED_VERSION..."
        pip install --force-reinstall transformers==$REQUIRED_VERSION
        echo "Reinstalled transformers==$REQUIRED_VERSION"
    else
        echo "Aborted. Patches require transformers==$REQUIRED_VERSION"
        exit 1
    fi
fi

PATCHES_DIR="$PROJECT_ROOT/third_party/transformers_patches"
if [ ! -d "$PATCHES_DIR" ]; then
    echo "Error: patches directory not found at $PATCHES_DIR"
    exit 1
fi

echo ""
echo "   Source: $PATCHES_DIR"
echo "   Target: $SITE_PACKAGES/transformers"

# Copy patches
cp -rv "$PATCHES_DIR"/* "$SITE_PACKAGES/transformers/"

echo ""
echo "Transformers patches applied successfully!"
echo ""
echo "WARNING: These patches permanently modify your transformers installation."
echo "   To revert, reinstall transformers:"
echo "   pip install --force-reinstall transformers==$REQUIRED_VERSION"

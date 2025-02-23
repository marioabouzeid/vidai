if ! command -v poetry &> /dev/null
then
    echo "Poetry could not be found, installing poetry"
    curl -sSL https://install.python-poetry.org | python3 -
    # add poetry to mac zshrc before continuing
    touch ~/.zshrc
    echo "export PATH=\"$HOME/.poetry/bin:$PATH\"" >> ~/.zshrc
    source ~/.zshrc
    echo "Poetry installed"
fi
poetry install --no-root

mkdir -p lib
if [ ! -f "lib/kokoro-v1.0.onnx" ] || [ ! -f "lib/voices-v1.0.bin" ]; then
    curl https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx -o lib/kokoro-v1.0.onnx
    curl https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -o lib/voices-v1.0.bin
fi

echo "Setup complete, add videos to lib/video/{theme} and music tracks to lib/audio"
echo "Modify the subject and theme data in vidal.py then run 'poetry run python vidal.py' to start the program"
# OCR API Testing Lab

Interactive Streamlit environment for testing and comparing multiple OCR APIs for receipt scanning.

## Features

- **6 OCR Provider Support**: Veryfi, Taggun, Klippa, OpenAI (ChatGPT), Mistral AI, Google Gemini
- **Runtime Configuration**: Switch between providers and adjust settings without code changes
- **Side-by-Side Comparison**: Process images with multiple providers simultaneously
- **Standardized Output**: Unified receipt data format across all providers
- **Metrics Dashboard**: Compare accuracy, speed, and cost
- **Export Functionality**: JSON, CSV, and BigQuery integration

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API credentials:**
   ```bash
   cp .env.template .env
   # Edit .env with your API keys
   ```

   **Note:** The `.env` file should be in the project root directory (not in `streamlit_app/`).

3. **Run the application:**
   ```bash
   # From the project root directory
   streamlit run streamlit_app/app.py
   ```

   **Important:** Always run the app from the project root directory to ensure environment variables load correctly.

## Project Structure

```
streamlit_app/
├── app.py                   # Main Streamlit application
├── config/                  # Configuration files
├── providers/               # OCR provider implementations
├── core/                    # Core business logic
├── ui/                      # UI components
├── utils/                   # Utilities
└── tests/                   # Test suite
```

## Usage

1. Select OCR providers from the sidebar
2. Configure each provider's settings
3. Upload a receipt image
4. Click "Process" to run OCR
5. Compare results side-by-side
6. Export data for analysis

## Deployment to Streamlit Cloud

### Prerequisites
- GitHub account
- Streamlit Cloud account (sign up at [share.streamlit.io](https://share.streamlit.io))
- API keys for the OCR providers you want to use

### Deployment Steps

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set main file path: `streamlit_app/app.py`
   - Click "Deploy"

3. **Configure Secrets:**
   - In your Streamlit Cloud app dashboard, go to "Settings" → "Secrets"
   - Copy the contents from `.streamlit/secrets.toml.example`
   - Paste and fill in your actual API keys:
   ```toml
   # Veryfi Configuration
   VERYFI_CLIENT_ID = "your_actual_client_id"
   VERYFI_CLIENT_SECRET = "your_actual_client_secret"
   VERYFI_USERNAME = "your_actual_username"
   VERYFI_API_KEY = "your_actual_api_key"

   # OpenAI Configuration
   OPENAI_API_KEY = "sk-..."

   # Mistral Configuration
   MISTRAL_API_KEY = "your_actual_api_key"

   # Gemini Configuration
   GEMINI_API_KEY = "your_actual_api_key"

   # Add other providers as needed
   ```
   - Click "Save"

4. **Verify Deployment:**
   - Your app will automatically redeploy with the new secrets
   - Click "Load from Environment" in the sidebar to load API keys
   - Test with a sample receipt image

### Local Testing with Streamlit Secrets

To test Streamlit secrets locally:

1. Create `.streamlit/secrets.toml` (gitignored):
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   # Edit and add your API keys
   ```

2. Run the app:
   ```bash
   streamlit run streamlit_app/app.py
   ```

### Troubleshooting Deployment

**Issue: "Module not found" errors**
- Ensure `requirements.txt` includes all dependencies
- Check that package versions are compatible

**Issue: "Missing API keys"**
- Verify secrets are properly configured in Streamlit Cloud settings
- Check that variable names match exactly (case-sensitive)
- Click "Load from Environment" button after adding secrets

**Issue: "App not updating"**
- Streamlit Cloud auto-deploys on git push
- Manually reboot from app settings if needed
- Check the logs for detailed error messages

## Related

Linear Issue: [DRT-3995](https://linear.app/drtail/issue/DRT-3995)

## License

Proprietary - Dr.Tail Inc.

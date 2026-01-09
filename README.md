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

## Related

Linear Issue: [DRT-3995](https://linear.app/drtail/issue/DRT-3995)

## License

Proprietary - Dr.Tail Inc.

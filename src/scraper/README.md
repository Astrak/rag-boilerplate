# Improved Article Scraper

This is a completely refactored and improved version of the original `scraper.py` file. The new architecture provides better maintainability, error handling, and modularity.

## 🚀 Key Improvements

### 1. **Modular Architecture**
- **Separation of Concerns**: Split the monolithic scraper into focused, single-responsibility classes
- **Better Testability**: Each component can be tested independently
- **Easier Maintenance**: Changes to one component don't affect others

### 2. **Configuration Management**
- **Environment Variables**: All settings configurable via environment variables
- **Centralized Config**: Single `ScraperConfig` class for all settings
- **Flexible Defaults**: Sensible defaults with easy override capability

### 3. **Proper Logging**
- **Structured Logging**: Replaced print statements with proper logging
- **File & Console Output**: Logs saved to file and displayed in console
- **Log Levels**: Different log levels for different types of information

### 4. **Error Handling & Resilience**
- **Retry Logic**: Automatic retry with exponential backoff for failed operations
- **Specific Exceptions**: Better error categorization and handling
- **Graceful Degradation**: System continues working even when some operations fail

### 5. **Type Safety**
- **Type Hints**: Full type annotations throughout the codebase
- **Data Models**: Proper data classes for articles and results
- **Validation**: Input validation and data integrity checks

### 6. **Performance Improvements**
- **Better Rate Limiting**: Configurable delays and polite scraping
- **Memory Management**: Streaming processing for large datasets
- **Checkpointing**: Resume long-running operations from where they left off

## 📁 New File Structure

```
app/scraper/
├── __init__.py              # Package initialization and exports
├── config.py                # Configuration management
├── models.py                # Data models and structures
├── utils.py                 # Utility functions and helpers
├── data_manager.py          # Data persistence and loading
├── web_scraper.py          # Web crawling and article extraction
├── text_processor.py       # Text chunking and document preparation
├── vector_store.py         # Vector store and embedding management
├── scraper.py              # Main orchestrator class
├── example_usage.py        # Usage examples and demonstrations
└── README.md               # This documentation
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCRAPER_DELAY` | `0.05` | Delay between requests (seconds) |
| `SCRAPER_MAX_WORKERS` | `3` | Number of concurrent scraping threads |
| `SCRAPER_TIMEOUT` | `15` | HTTP request timeout (seconds) |
| `OPENAI_MODEL` | `text-embedding-3-large` | OpenAI embedding model |
| `MAX_TOKENS_PER_REQUEST` | `260000` | Max tokens per OpenAI API call |
| `CHECKPOINT_DIR` | `polemia-embeddings` | Directory for checkpoint files |
| `SCRAPED_ARTICLES_FILE` | `./scraped_articles.pkl.gz` | Articles storage file |
| `VECTORSTORE_DIR` | `./vectorstore` | Vector store directory |
| `CHUNK_SIZE` | `2600` | Text chunk size for processing |
| `CHUNK_OVERLAP` | `500` | Overlap between text chunks |
| `FAISS_CHUNK_SIZE` | `20` | FAISS index chunk size |
| `SCRAPER_MAX_RETRIES` | `3` | Maximum retry attempts |
| `SCRAPER_RETRY_DELAY` | `1.0` | Base delay between retries |

## 🚀 Usage Examples

### Basic Usage

```python
from scraper import ArticleScraper, ScraperConfig

# Create configuration
config = ScraperConfig(
    delay=0.1,
    max_workers=2,
    chunk_size=2000
)

# Initialize scraper
scraper = ArticleScraper(
    base_url="https://example.com",
    excluded_paths=["/admin", "/private"],
    config=config
)

# Run full pipeline
result = scraper.run_full_pipeline()
print(f"Pipeline result: {result}")
```

### Step-by-Step Usage

```python
# Just discover and scrape
scraping_result = scraper.discover_and_scrape()
print(f"Success rate: {scraping_result.success_rate:.1%}")

# Create vector store from existing articles
vector_store = scraper.create_vector_store_from_existing()

# Search for similar documents
similar_docs = scraper.search_similar_documents("What is machine learning?")
```

### Advanced Features

```python
# Checkpointed embeddings for large datasets
scraper.create_embeddings_with_checkpoint()

# Get system statistics
stats = scraper.get_statistics()
print(f"Articles: {stats['articles_count']}")

# Clean up old data
scraper.cleanup_old_data(keep_recent_backups=3)
```

## 🔍 Component Details

### ArticleScraper (Main Class)
- **Purpose**: Main orchestrator that coordinates all operations
- **Features**: Pipeline management, error handling, statistics
- **Methods**: `run_full_pipeline()`, `discover_and_scrape()`, `search_similar_documents()`

### WebScraper
- **Purpose**: Handles web crawling and article extraction
- **Features**: URL discovery, polite scraping, multiple selector fallbacks
- **Methods**: `discover_urls()`, `scrape_articles()`, `scrape_article()`

### TextProcessor
- **Purpose**: Manages text chunking and document preparation
- **Features**: Smart text splitting, token counting, batch creation
- **Methods**: `split_article_text()`, `create_batches_for_embeddings()`

### VectorStoreManager
- **Purpose**: Handles embeddings and FAISS operations
- **Features**: Checkpointing, chunked indices, similarity search
- **Methods**: `create_vector_store()`, `create_embeddings_with_checkpoint()`

### DataManager
- **Purpose**: Manages data persistence and loading
- **Features**: Backup management, file validation, size monitoring
- **Methods**: `save_articles()`, `load_articles()`, `backup_articles()`

## 🛡️ Error Handling

### Retry Logic
- Automatic retry with exponential backoff
- Configurable retry attempts and delays
- Graceful handling of temporary failures

### Data Validation
- Input validation for URLs and content
- File integrity checks
- Fallback mechanisms for missing data

### Logging
- Comprehensive error logging
- Progress tracking
- Debug information for troubleshooting

## 📊 Monitoring & Statistics

### Progress Tracking
- Real-time progress updates
- Success/failure rate monitoring
- ETA calculations for long operations

### System Health
- File size monitoring
- Memory usage tracking
- Component status reporting

### Performance Metrics
- Processing speed measurements
- Resource utilization tracking
- Bottleneck identification

## 🔄 Migration from Old Version

### Breaking Changes
- **Import Changes**: New modular import structure
- **Configuration**: Environment variables instead of hardcoded constants
- **Method Names**: Some method names have changed for clarity

### Migration Steps
1. **Update Imports**: Use new package structure
2. **Set Environment Variables**: Configure via environment or config file
3. **Update Method Calls**: Use new method names and signatures
4. **Test**: Verify functionality with new architecture

### Compatibility
- **Data Files**: Existing data files are still compatible
- **API**: Core functionality remains the same
- **Output**: Results format is unchanged

## 🧪 Testing

### Unit Tests
- Each component can be tested independently
- Mock objects for external dependencies
- Comprehensive test coverage

### Integration Tests
- End-to-end pipeline testing
- Error condition testing
- Performance benchmarking

### Example Usage
```bash
# Run example script
python -m app.scraper.example_usage

# Run with custom configuration
SCRAPER_DELAY=0.2 python -m app.scraper.example_usage
```

## 🚨 Best Practices

### Rate Limiting
- Always respect website robots.txt
- Use appropriate delays between requests
- Monitor for rate limiting responses

### Error Handling
- Implement proper retry logic
- Log all errors for debugging
- Gracefully handle partial failures

### Data Management
- Regular backups of scraped data
- Validate data integrity
- Clean up old files periodically

### Performance
- Monitor memory usage
- Use checkpointing for large datasets
- Optimize batch sizes for your use case

## 🤝 Contributing

### Code Style
- Follow PEP 8 guidelines
- Use type hints throughout
- Add docstrings to all methods

### Testing
- Write tests for new features
- Ensure backward compatibility
- Test error conditions

### Documentation
- Update README for new features
- Add inline code comments
- Provide usage examples

## 📝 License

This improved scraper maintains the same license as the original project.

## 🆘 Support

### Common Issues
- **Memory Issues**: Reduce batch sizes or use checkpointing
- **Rate Limiting**: Increase delays between requests
- **API Errors**: Check OpenAI API key and quotas

### Debugging
- Enable debug logging for detailed information
- Check log files for error details
- Use statistics methods to monitor progress

---

**Note**: This is a significant refactor that improves maintainability, reliability, and performance while maintaining backward compatibility with existing data and functionality.

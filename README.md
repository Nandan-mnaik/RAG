# Document Q&A System with Feedback Learning (RAG)

A sophisticated **Retrieval-Augmented Generation (RAG)** system that processes PDF documents and learns from user feedback to continuously improve response quality and relevance.

## 🌟 Features

- **📄 PDF Document Processing**: Automatic loading and chunking of PDF documents
- **🔍 Semantic Search**: Advanced vector-based document retrieval using ChromaDB
- **🤖 LLM Integration**: Powered by Llama 3.2 for natural language understanding
- **📊 User Feedback Learning**: Collects and applies user feedback to improve system performance
- **⚡ Dynamic Index Enhancement**: Automatically fine-tunes retrieval based on feedback data
- **📈 Performance Analytics**: Tracks relevance and quality metrics over time
- **🔄 Iterative Improvement**: Continuously evolves based on user interactions

## 🛠️ Technology Stack

- **Language Model**: Llama 3.2 (via Ollama)
- **Vector Database**: ChromaDB with sentence-transformers
- **Framework**: LangChain for RAG pipeline
- **Document Processing**: PyPDF for PDF handling
- **Embeddings**: Sentence-transformers for semantic understanding
- **Environment**: Python 3.8+

## 📋 Prerequisites

Before running the system, ensure you have:

1. **Python 3.8 or higher** installed
2. **Ollama** installed and running locally
3. **Llama 3.2 model** pulled in Ollama: `ollama pull llama3.2:1b`
4. **PDF documents** to process (sample included)

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd rag-feedback-system
```

### 2. Install Dependencies
```bash
pip install langchain langchain-community python-dotenv chromadb sentence-transformers pypdf tf-keras
```

### 3. Setup Ollama
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the Llama model
ollama pull llama3.2:1b
```

### 4. Run the System
```bash
jupyter notebook RAG_Enhanced.ipynb
```

## 📁 Project Structure

```
rag-feedback-system/
├── RAG_Enhanced.ipynb          # Main notebook with documentation
├── RAG.ipynb                   # Original implementation
├── README.md                   # This file
├── data/                       # PDF documents directory
├── feedback_data.json          # User feedback storage
└── RAG_TECHNIQUES/             # Helper functions module
    └── helper_functions.py     # Utility functions
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:
```env
KMP_DUPLICATE_LIB_OK=TRUE
OLLAMA_BASE_URL=http://localhost:11434
```

### Model Configuration
The system uses Llama 3.2:1b by default. To use a different model:
```python
llm = ChatOllama(
    model="llama3.2:3b",  # Change model size
    temperature=0,
)
```

## 📖 Usage Guide

### Basic Usage
1. **Initialize the System**: Run cells 1-11 to set up the RAG pipeline
2. **Load Documents**: The system automatically processes PDF documents in the `data/` directory
3. **Ask Questions**: Use the query interface to ask questions about your documents
4. **Provide Feedback**: Rate responses (1-5) for relevance and quality
5. **See Improvements**: The system learns and improves from your feedback

### Advanced Features

#### Custom PDF Processing
```python
# Process a specific PDF
qa_chain, retriever, content, current_pdf = initialize_rag_system("path/to/your/document.pdf")
```

#### Feedback Integration
```python
# Provide structured feedback
feedback = get_user_feedback(
    query="Your question",
    response="System response", 
    relevance=4,  # 1-5 scale
    quality=5,    # 1-5 scale
    comments="Helpful but could be more detailed"
)
```

#### Performance Analytics
```python
# View feedback statistics
feedback_data = load_feedback_data()
avg_relevance = sum(f['relevance'] for f in feedback_data) / len(feedback_data)
print(f"Average relevance score: {avg_relevance:.2f}")
```

## 🔍 How It Works

### 1. Document Processing
- PDFs are loaded and split into semantic chunks
- Text chunks are embedded using sentence-transformers
- Embeddings are stored in ChromaDB vector database

### 2. Query Processing
- User queries are embedded using the same model
- Semantic similarity search retrieves relevant documents
- Context is provided to Llama 3.2 for response generation

### 3. Feedback Learning
- User ratings and comments are collected and stored
- System analyzes feedback relevance using LLM
- High-quality Q&A pairs are integrated into the knowledge base
- Retrieval scores are dynamically adjusted based on feedback

### 4. Continuous Improvement
- The system tracks performance metrics over time
- Feedback data informs future retrieval and ranking decisions
- Index is fine-tuned with proven high-quality responses

## 📊 Performance Metrics

The system tracks several key metrics:
- **Response Relevance**: User-rated relevance scores (1-5)
- **Response Quality**: User-rated quality scores (1-5)
- **Feedback Volume**: Number of feedback entries collected
- **Improvement Trends**: Performance changes over time

## 🔧 Customization

### Adding New Documents
```python
# Place PDF files in the data/ directory
# Or specify custom path:
qa_chain, retriever, content, current_pdf = initialize_rag_system("custom/path/document.pdf")
```

### Adjusting Retrieval Settings
```python
# Modify chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Increase for longer chunks
    chunk_overlap=200  # Adjust overlap
)
```

### Changing LLM Parameters
```python
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.1,  # Adjust creativity
    top_p=0.9,        # Control randomness
)
```

## 🚨 Troubleshooting

### Common Issues

**Ollama Connection Error**
```bash
# Ensure Ollama is running
ollama serve
```

**Model Not Found**
```bash
# Pull the required model
ollama pull llama3.2:1b
```

**Memory Issues**
```python
# Use smaller model or reduce batch size
llm = ChatOllama(model="llama3.2:1b")  # Instead of 3b or 7b
```

**PDF Loading Errors**
- Ensure PDF files are not password-protected
- Check file permissions and accessibility
- Verify PDF format compatibility

## 📈 Future Enhancements

- [ ] **Web Interface**: Streamlit/Gradio web UI
- [ ] **Multi-document Support**: Process multiple PDFs simultaneously
- [ ] **Advanced Analytics**: Detailed performance dashboards
- [ ] **Export Capabilities**: Export feedback data and metrics
- [ ] **Model Flexibility**: Support for different LLM providers
- [ ] **Batch Processing**: Handle large document collections

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for the RAG framework
- **Ollama** for local LLM serving
- **ChromaDB** for vector storage
- **Sentence-Transformers** for embeddings
- **Meta** for the Llama models

## 📞 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the notebook documentation

---

**Built with ❤️ for intelligent document processing and continuous learning**

**Note: I have added the enhanced version for better readability and explanation of code**

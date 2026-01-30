# Copyright (C) 2025, Advanced Micro Devices, Inc.  All rights reserved.
#
# Author: Preston Walker, AMD
# SPDX-License-Identifier: MIT

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings    
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter  
from llama_index.core.schema import Document, TextNode    
import os    
import re    
    
class MultiContentSplitter:    
    """A text splitter that handles multiple content types intelligently."""    
        
    def __init__(self):    
        # Use regular text splitters instead of CodeSplitter  
          
        # For code: Use token splitter with newline separation  
        self.code_splitter = TokenTextSplitter(    
            chunk_size=512,          
            chunk_overlap=50,        
            separator="\n"         # Split on newlines which works well for code  
        )    
            
        self.text_splitter = SentenceSplitter(    
            chunk_size=512,         
            chunk_overlap=50        
        )    
            
        self.token_splitter = TokenTextSplitter(    
            chunk_size=256,         
            chunk_overlap=20    
        )  
        
    # Rest of the class remains the same  
    def split_documents(self, documents):    
        """Split documents based on their content type."""    
        all_nodes = []    
            
        for doc in documents:    
            # Determine document type based on extension and content    
            doc_type = self._detect_document_type(doc)    
                
            # Apply the appropriate splitter    
            if doc_type == "code":    
                nodes = self._split_code_document(doc)    
            elif doc_type == "text":    
                nodes = self._split_text_document(doc)    
            else:  # "mixed" or unknown    
                nodes = self._split_mixed_document(doc)    
                    
            all_nodes.extend(nodes)    
                
        return all_nodes    
        
    def _detect_document_type(self, document):    
        """Detect the type of document based on extension and content."""    
        # Check file extension first    
        file_path = document.metadata.get("file_path", "")    
        file_ext = os.path.splitext(file_path)[1].lower()    
            
        # Code file extensions    
        code_extensions = [".cpp", ".h", ".c", ".hpp", ".cc"]    
        if file_ext in code_extensions:    
            return "code"    
            
        # Text/document extensions    
        text_extensions = [".txt", ".md", ".tex", ".rst", ".pdf"]    
        if file_ext in text_extensions:    
            return "text"    
            
        # If extension doesn't clearly indicate type, analyze content    
        content = document.text    
            
        # Check for code indicators    
        code_indicators = [    
            "#include", "pragma", "void ", "int ", "class ", "struct ",    
            "{", "}", "for(", "while(", "if(", "else", "return "    
        ]    
            
        code_indicator_count = sum(1 for indicator in code_indicators if indicator in content)    
            
        # If many code indicators are present, treat as code    
        if code_indicator_count >= 3:    
            return "code"    
            
        # Default to text for most content    
        return "text"    
        
    def _split_code_document(self, document):    
        """Split a code document using the code-specific splitter."""    
        # Create temp Document objects as needed by CodeSplitter    
        temp_doc = Document(text=document.text, metadata=document.metadata)    
        return self.code_splitter.get_nodes_from_documents([temp_doc])    
        
    def _split_text_document(self, document):    
        """Split a text document using the sentence splitter."""    
        temp_doc = Document(text=document.text, metadata=document.metadata)    
        return self.text_splitter.get_nodes_from_documents([temp_doc])    
        
    def _split_mixed_document(self, document):    
        """Split a mixed document using a more conservative approach."""    
        temp_doc = Document(text=document.text, metadata=document.metadata)    
        return self.token_splitter.get_nodes_from_documents([temp_doc])    

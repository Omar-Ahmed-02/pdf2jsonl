document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const statusDiv = document.getElementById('status');
    const downloadLinkContainer = document.getElementById('download-link-container');
    let selectedFile = null;

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type === 'application/pdf') {
            fileInput.files = files;
            selectedFile = files[0];
            updateDropZoneText();
        } else {
            alert('Please drop a PDF file.');
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            selectedFile = fileInput.files[0];
            updateDropZoneText();
        }
    });
    
    function updateDropZoneText() {
        if (selectedFile) {
            dropZone.querySelector('p').textContent = `Selected file: ${selectedFile.name}`;
        } else {
            dropZone.querySelector('p').textContent = 'Drag & Drop PDF here or click to select';
        }
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!selectedFile) {
            alert('Please select a file first.');
            return;
        }

        statusDiv.textContent = 'Processing... This may take a moment.';
        downloadLinkContainer.innerHTML = '';

        const formData = new FormData(form);
        // formData.append('file', selectedFile); // The file is already in the form's file input

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                statusDiv.textContent = 'Processing complete!';
                const downloadLink = document.createElement('a');
                downloadLink.href = result.download_url;
                downloadLink.textContent = 'Download JSONL File';
                downloadLink.setAttribute('download', result.filename);
                downloadLinkContainer.appendChild(downloadLink);
            } else {
                statusDiv.textContent = `Error: ${result.error}`;
            }
        } catch (error) {
            statusDiv.textContent = 'An unexpected error occurred. Please check the console.';
            console.error('Error during upload:', error);
        }
    });
}); 
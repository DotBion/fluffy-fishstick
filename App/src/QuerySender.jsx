// QuerySender.js
import React, { useState } from 'react';

export default function QuerySender({ message }) {
  const [responseHtml, setResponseHtml] = useState('');

  const sendQuery = async () => {
    if (!message) return;
    try {
      const res = await fetch('http://localhost:5050/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: message }),
      });
      const data = await res.json();
      setResponseHtml(data.response);       // response now contains HTML <p>…</p> blocks
      console.log('Response HTML:', data.response);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="query-sender">
      <button onClick={sendQuery}>Send Query</button>

      {/* Render the HTML we received from the backend */}
      <div
        className="chat-history"
        dangerouslySetInnerHTML={{ __html: responseHtml }}
      />
    </div>
  );
}

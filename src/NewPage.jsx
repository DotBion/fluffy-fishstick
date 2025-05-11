// NewPage.js

import React, { useState } from 'react';
import './NewPage.css';
import QuerySender from './QuerySender';


function NewPage() {
  const [selectedStock, setSelectedStock] = useState('');
  const [message, setMessage] = useState('');

  const handleStockChange = e => setSelectedStock(e.target.value);
  const handleMessageChange = e => setMessage(e.target.value);
  const sendMessage = () => {
    console.log("Selected Stock:", selectedStock);
    console.log("Message:", message);
    // …your send logic…
  };

  const nifty50Stocks = [ /* …your list… */ ];

  return (
    <div className="NewPage">
      <div className="left-section">
        <select value={selectedStock} onChange={handleStockChange}>
          <option value="">Choose from Nifty-50</option>
          {nifty50Stocks.map(stock => (
            <option key={stock} value={stock}>
              {stock}
            </option>
          ))}
        </select>
      </div>

      <div className="chat-section">
        {/* Render responses here */}
        <div className="chat-history">
          <QuerySender message={message} />
        </div>

        {/* Text input (Enter key can trigger send) */}
        <div className="text-input">
          <input
            type="text"
            placeholder="Type your message..."
            value={message}
            onChange={handleMessageChange}
            onKeyDown={e => e.key === 'Enter' && sendMessage()}
          />
        </div>
      </div>
    </div>
  );
}

export default NewPage;

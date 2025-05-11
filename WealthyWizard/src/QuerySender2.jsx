// QuerySender.js
import React, { useState } from 'react';

function QuerySender2({ message }) { 
    console.log("B2", message);
  const sendQuery = async () => {
    try {
      const res = await fetch('http://localhost:5050/api/com_name', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: message }), // Use message prop as the query
      });
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div>
      {/* Removed input for query */}
      <button onClick={sendQuery}>Select Stock</button>
      
    </div>
  );
}

export default QuerySender2;

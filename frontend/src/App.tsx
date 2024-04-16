import React, { useEffect, useState } from "react";
import "./App.css";

function App() {
  const [data, setdata] = useState({
    name: "",
    date: "",
    occupation: "",
  });

  useEffect(() => {
    fetch("/data").then((res) =>
      res.json().then((data) =>
        setdata({
          name: data.name,
          date: data.date,
          occupation: data.occupation,
        }),
      ),
    );
  }, []);

  return (
    <div className="App">
      <p>{data.name}</p>
      <p>{data.date}</p>
      <p>{data.occupation}</p>
    </div>
  );
}

export default App;

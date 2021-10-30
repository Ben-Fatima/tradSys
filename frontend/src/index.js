import React from "react"
import ReactDOM from "react-dom"
import "./index.css"
import App from "./App"

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById("root")
)

/*function Counter() {
  const [value, setValue] = React.useState(0)
  setValue(value + 1)
  return value
}
console.log(Counter())
console.log(Counter())
console.log(Counter())*/

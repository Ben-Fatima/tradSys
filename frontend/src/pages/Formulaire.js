import axios from "axios"
import { AnimatePresence, motion } from "framer-motion"
import React, { useRef, useState } from "react"

export function Alert({ children }) {
  return (
    <div className="pt-2">
      <div className="bg-red-200 border-red-400 rounded text-red-900 p-2 mb-4 m-auto shadow-md" role="alert">
        {children}
      </div>
    </div>
  )
}
const Formulaire = () => {
  const [page, setPage] = useState("form")
  return (
    <div className="w-full">
      <div className="w-9/12 m-auto p-4">
        <AnimatePresence exitBeforeEnter>
          {page === "form" && <Form setPage={setPage} />}
          {page === "loading" && <Loading />}
          {page === "result" && <Result setPage={setPage} />}
        </AnimatePresence>
      </div>
    </div>
  )
}
export default Formulaire

const Form = ({ setPage, value }) => {
  const dataName = useRef()
  const methodName = useRef()
  const h_dimName = useRef()
  const n_layersName = useRef()
  const lrName = useRef()
  const n_epoquesName = useRef()
  const horizonName = useRef()
  const archName = useRef()
  const algoName = useRef()
  const [mthd, setMthd] = useState("")
  const [error, setError] = useState(null)
  const upload = async (e) => {
    e.preventDefault()
    const data = dataName.current.value
    const method = methodName.current.value
    var arch = null
    var h_dim = null
    var n_layers = null
    var lr = null
    var n_epoques = null
    var horizon = null
    var algo = null
    if (method === "train") {
      arch = archName.current.value
      h_dim = h_dimName.current.value
      n_layers = n_layersName.current.value
      lr = lrName.current.value
      n_epoques = n_epoquesName.current.value
      horizon = horizonName.current.value
    } else if (method === "pred") {
      algo = algoName.current.value
    }
    const validate = (value) => {
      if (value === "") {
        throw `Please fill the data name`
      }
    }
    try {
      validate(dataName.current.value)
    } catch (error) {
      setError(error)
      return
    }
    setPage("loading")
    const response = await axios.post(`http://localhost:8000/compute`, {
      algo: algo,
      data: data,
      method: method,
      arch: arch,
      h_dim: h_dim,
      n_layers: n_layers,
      lr: lr,
      n_epoques: n_epoques,
      horizon: horizon,
    })
    console.log(response)
    setPage("result")
  }
  return (
    <div className="w-full m-auto">
      <AnimatePresence exitBeforeEnter>
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
          <div className="w-full m-auto" id="ensb">
            <form onSubmit={upload}>
              <div className="w-1/4 m-auto pt-10 narrow">
                <label className="block">Enter your data name:</label>
                <input className="rounded bg-gray-100 text-black p-1 border" ref={dataName}></input>
                {error && <Alert>{error}</Alert>}
              </div>
              <div className="w-1/4 m-auto  narrow">
                <label>Select the method you want:</label>
                <select
                  className="narrow custom-select"
                  ref={methodName}
                  onChange={(e) => {
                    setMthd(methodName.current.value)
                  }}
                >
                  <option value="" selected>
                    --Please choose an option--
                  </option>
                  <option value="train">Train</option>
                  <option value="pred">Predict</option>
                </select>
              </div>
              {mthd === "train" && (
                <div>
                  <div className="w-1/4 m-auto  narrow">
                    <label>select the architecture:</label>
                    <select className="narrow custom-select" ref={archName}>
                      <option value="LSTM">LSTM</option>
                      <option value="GRU" selected>
                        GRU
                      </option>
                    </select>
                  </div>
                  <div className="w-1/4 m-auto  narrow">
                    <label>select the number of the hidden dimensions:</label>
                    <select ref={h_dimName} className="narrow custom-select">
                      <option value="16" selected>
                        16
                      </option>
                      <option value="32">32</option>
                      <option value="64">64</option>
                    </select>
                  </div>
                  <div className="w-1/4 m-auto  narrow">
                    <label>select the number of layers:</label>
                    <select ref={n_layersName} className="narrow custom-select">
                      <option value="1" selected>
                        1
                      </option>
                      <option value="2">2</option>
                      <option value="3">3</option>
                    </select>
                  </div>
                  <div className="w-1/4 m-auto  narrow">
                    <label>select the learning rate:</label>
                    <select ref={lrName} className="narrow custom-select">
                      <option value="0.05" selected>
                        0.05
                      </option>
                      <option value="0.01">0.01</option>
                      <option value="0.001">0.001</option>
                    </select>
                  </div>
                  <div className="w-1/4 m-auto  narrow">
                    <label>select the number of the epoques:</label>
                    <select ref={n_epoquesName} className="narrow custom-select">
                      <option value="50" selected>
                        50
                      </option>
                      <option value="100">100</option>
                      <option value="150">150</option>
                    </select>
                  </div>
                  <div className="w-1/4 m-auto  narrow">
                    <label>select the horizon:</label>
                    <select ref={horizonName} className="narrow custom-select">
                      <option value="50" selected>
                        50
                      </option>
                      <option value="100">100</option>
                      <option value="150">150</option>
                    </select>
                  </div>
                </div>
              )}
              {mthd === "pred" && (
                <div className="w-1/4 m-auto  narrow">
                  <label>select the algorithm:</label>
                  <select ref={algoName} className="narrow custom-select">
                    <option value="dmac" selected>
                      dmac
                    </option>
                    <option value="obv">obv</option>
                    <option value="alligator">alligator</option>
                    <option value="Rsi">Rsi</option>
                  </select>
                </div>
              )}
              <div className="w-1/4 m-auto  narrow">
                <button
                  type="submit"
                  className="mt-4 inline-flex  items-center font-semibold btn3 py-2 px-4 font-small rounded pro"
                >
                  <i class="far fa-check-circle"></i>Submit
                </button>
              </div>
            </form>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  )
}
const Loading = () => {
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
      <div className="w-full m-auto text-center">
        <div className="pt-24">
          <h1>Preparing your result...</h1>
          <div class="lds-dual-ring"></div>
        </div>
      </div>
    </motion.div>
  )
}
const Result = ({ setPage }) => {
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
      <div>
        <img className="w-full m-auto pt-10" src="http://localhost:8000/figure.png" alt="result"></img>
        <button
          onClick={() => setPage("form")}
          className="mt-4 inline-flex  items-center font-semibold btn3 py-2 px-4 font-small rounded pro"
        >
          <i class="fas fa-arrow-circle-left p-1"></i>Back
        </button>
      </div>
    </motion.div>
  )
}

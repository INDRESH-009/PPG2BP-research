//Destrcturing in Array of Objects 

const people = [
    {id:1,name:"Alice",city:"NY"},
    {id:2,name:"Bob",city:"LA"}
]

//without destructuring
people.forEach(p=>{
    console.log(`${p.name} lives in ${p.city}`)
})

//with destrcturing
people.forEach(({name,city})=>{
    console.log(`${name} lives in ${city}`)
})
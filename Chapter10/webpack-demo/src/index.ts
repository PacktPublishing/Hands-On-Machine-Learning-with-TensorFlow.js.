import {Message} from './message';

const message = new Message();
const p = document.getElementById("message");
p.innerText = message.say();